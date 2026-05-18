import copy
from collections import Counter
import os
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.signal import decimate, resample, savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.special import betainc
import scipy.constants as consts

# STOLEN FROM LARCH HEHE
# EINS_FACTOR  = hbarc*hbarc/(2 * k_boltz * amu) = 24.254360157751783 Ang^2 * K * amu
#    k_boltz = 8.6173324e-5  # [eV / K]
#    amu     = 931.494061e6  # [eV / (c*c)]
#    hbarc   = 1973.26938    # [eV * A]
EINS_FACTOR = 1.e20*consts.hbar**2/(2*consts.k*consts.atomic_mass)

import larch.io as lio
import larch.xafs as lx

from larch import Group
from larch.fitting import param, guess, param_group
from larch.math import interp1d, remove_dups

import warnings

from .extra import Plot


class Exafs:
    def __init__(
        self,
        file=None,
        df=None,
        **kwargs
    ):
        '''
        KWARGS:
            file (str or pathlib.Path): path to file
            df (pd.DataFrame): DataFrame to parse (df.columns = ['Energy (eV)', 'Intensity', 'Ref Intensity'], 'Ref Intensity' is optional)
            file (str): path to file (df is always preferred if both df and file are specified)
        df should have columns ['Energy (eV)', 'Intensity', 'Ref Intensity']
        if both df and file are specified, df takes precedence
        '''
        if df is not None:
            self.df = df
        elif file is not None:
            self.file = pathlib.Path(file)
            self.df = self.read_csv(self.file, **kwargs)
        
    def read_csv(
        self,
        file,
        format='SPring-8',
        skiprows=13,
        *args,
        **kwargs
    ):
        df = pd.read_csv(
            file,
            sep=r"\s+",
            skiprows=skiprows,
        )
        if format == 'SPring-8':
            df.columns = ['Angle (c)', 'Angle (o)', 'Time (s)', 'I0', 'I1']
            # attempt to auto-detect d-spacing of monochromator grating
            title = pd.read_csv(file, skiprows=4, nrows=0)
            title = title.columns.values[0]
            idx0 = title.find('D=')
            idx1 = title.find('A')
            d = float(title[idx0+2:idx1])
            # calculate energy from grating angle
            df['Energy (eV)'] = self.energy(df['Angle (o)'], d/10)
            # calculate absorption
            df['Intensity'] = self.intensity(df['I0'], df['I1'])
        if format == '2col':
            df.columns = ['Energy (eV)', 'Intensity']
            
        self.df = df
        return self.df
    
    def gete0(
        self,
        bounds,
        column='Energy (eV)',
        step=0.025,
        window_length=6,
        polyorder=3,
        grid_points=None,
        **kwargs
    ):
        df_reevaluated = self.interpolate_and_downsample(
            self.df, 
            column, 
            bounds, 
            step,
            grid_points=grid_points,
            **kwargs
        )
        dy = savgol_filter(df_reevaluated['Intensity'], window_length, polyorder, deriv=1, delta=step)
        idx = np.argmax(dy)
        self.e0 = df_reevaluated['Energy (eV)'].iloc[idx]
        return self.e0
    
    def rebin(
        self,
        E0=None,
        pre_edge_cutoff=-300,
        Emax=None,
        kmax=None,
        xanes_region=[-30, 50],
        pre_edge_step=10, # eV
        exafs_step=0.05, # 1/Å
        mode='decimate',
        s=0
    ):
        if E0 is None:
            E0 = self.e0
        self.df_orig = np.copy(self.df)
        # pre_edge = np.array(pre_edge) + E0
        # exafs = self.k_to_e(np.array(exafs)) + E0
        
        if pre_edge_cutoff + E0 < min(self.df['Energy (eV)']):
            pre_edge_cutoff = min(self.df['Energy (eV)']) - E0
        pre_edge = np.array([pre_edge_cutoff, xanes_region[0]]) + E0
        if Emax is None:
            Emax = self.df['Energy (eV)'].max() - E0
        elif Emax > (self.df['Energy (eV)'].max() - E0):
            Emax = self.df['Energy (eV)'].max() - E0
        print('Emax = ', Emax)
        print('kmax = ', self.E_to_k(Emax, 0))
        # if kmax is None:
        #     kmax = self.E_to_k(Emax, 0)
        #     print('kmax = {}'.format(kmax))
        # generate EXAFS grid in k-space
        if kmax is None:
            kmax = self.E_to_k(Emax, 0)
        exafs_grid = np.arange(0, kmax, exafs_step)
        # convert EXAFS grid to E-space
        exafs_grid = self.k_to_E(exafs_grid, E0)
        exafs_grid = exafs_grid[exafs_grid >= xanes_region[1] + E0]
        exafs_grid = exafs_grid[exafs_grid <= Emax + E0]
        
        # truncate below pre_edge[0]
        self.df[self.df['Energy (eV)'] >= pre_edge[0]]
        # truncate above exafs[1]
        self.df[self.df['Energy (eV)'] <= Emax + E0]
        
        if mode == 'spline':
            # interpolate downsample below pre_edge[1]
            self.df = pd.concat([
                self.interpolate_and_downsample(self.df[(self.df['Energy (eV)'] >= pre_edge[0]) & (self.df['Energy (eV)'] <= pre_edge[1])], 'Energy (eV)', pre_edge, pre_edge_step, s=s),
                self.df[self.df['Energy (eV)'] >= pre_edge[1]]
            ])
            # interpolate downsample above exafs[0]
            self.df = pd.concat([
                self.df[self.df['Energy (eV)'] <= xanes_region[1] + E0],
                self.interpolate_and_downsample(
                    self.df[(self.df['Energy (eV)'] >= xanes_region[1] + E0) & (self.df['Energy (eV)'] <= Emax + E0)], 
                    'Energy (eV)', 
                    [0,0], 
                    0, 
                    grid_points=exafs_grid,
                    s=s
                )
            ])
        elif mode == 'decimate':
            self.df = pd.concat([
                self.simple_decimate(
                    self.df[self.df['Energy (eV)'] <= pre_edge[1]],
                    'Energy (eV)', 
                    pre_edge, 
                    pre_edge_step
                ),
                self.df[self.df['Energy (eV)'] >= pre_edge[1]]
            ])
            self.df = pd.concat([
                self.df[self.df['Energy (eV)'] <= xanes_region[1] + E0],
                self.simple_decimate(
                    self.df[(self.df['Energy (eV)'] >= xanes_region[1] + E0) & (self.df['Energy (eV)'] <= Emax + E0)], 
                    'Energy (eV)', 
                    [0,0], 
                    0, 
                    grid_points=exafs_grid
                )
            ])
        
    def to_csv(
        self,
        target,
        sep=' ',
        header=False
    ):
        sub_df = self.df[["Energy (eV)", "Intensity"]]
        sub_df.to_csv(target, index=False, sep=sep, header=header)
        
    @staticmethod
    def bragg(theta, d, n=1):
        return n * 2 * d * np.sin(np.radians(theta))

    @classmethod
    def energy(cls, theta, d, n=1):
        return 1239.8 / cls.bragg(theta, d, n)

    @staticmethod
    def intensity(I0, I1):
        return np.log(I0 / I1)
    
    @staticmethod
    def E_to_k(
        E,
        E0
    ):
        return np.sqrt(E-E0) * 0.512
    
    @staticmethod
    def k_to_E(
        k,
        E0
    ):
        return (k/0.512)**2 + E0
    
    @staticmethod
    def simple_decimate(
        df,
        column,
        bounds, 
        step,
        grid_points=None
    ):
        if grid_points is None:
            grid_points = np.arange(bounds[0], bounds[1], step)
        df = df.sort_values(by=column)
        e = df[column].to_numpy()
        n = len(e)
        m = len(grid_points)
        diff_matrix = np.empty((n, m))
        for i in range(n):
            diff_matrix[i] = np.abs(grid_points - e[i])
        idx = np.argmin(diff_matrix, axis=0)
        return df.iloc[idx]
        

    
    @staticmethod
    def interpolate_and_downsample(
        df, 
        column, 
        bounds, 
        step,
        grid_points=None,
        s=0
    ):
        """
        Use splines to reevaluate a DataFrame on a specified grid for a given column.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to reevaluate.
        column (str): The column name to base the reevaluation on.
        grid_points (array-like): The grid points to reevaluate the DataFrame on.
        
        Returns:
        pd.DataFrame: The reevaluated DataFrame.
        """
        if grid_points is None:
            grid_points = np.arange(bounds[0], bounds[1], step)
        # Sort the DataFrame based on the specified column
        df_sorted = df.sort_values(by=column)
        
        # Create a dictionary to store the reevaluated data
        reevaluated_data = {column: grid_points}
        
        # Use splines to reevaluate each column in the DataFrame
        for col in df_sorted.columns:
            if col != column:
                spline = UnivariateSpline(df_sorted[column], df_sorted[col], s=s)
                reevaluated_data[col] = spline(grid_points)
        
        # Create a new DataFrame with the reevaluated data
        df_reevaluated = pd.DataFrame(reevaluated_data)
        
        return df_reevaluated
    

class Batch:
    def __init__(
        self,
        file_list,
        out_names,
        E0,
        file_extension='',
        out_extension='',
        rebin=True,
        plot=False,
        save=True,
        abort_at_error=False,
        **kwargs
    ):
        if plot:
            fig, ax = plt.subplots(layout='constrained')
        for i in range(len(file_list)):
            try:
                xafs = Exafs("{}{}".format(file_list[i], file_extension))
            except:
                warnings.warn("Error reading file: {}".format(file_list[i]))
                pass
            if rebin:
                try:
                    xafs.rebin(
                        E0,
                        **kwargs
                    )
                except:
                    warnings.warn("Error rebinning file: {}".format(file_list[i]))
                    pass
            if save:
                xafs.to_csv("{}{}".format(out_names[i], out_extension))
            if plot:
                ax.plot(xafs.df['Energy (eV)'], xafs.df['Intensity'], 'ko')
                plt.show()
                
                
class Rsxap:
    def __init__(
        self
    ):
        pass

    fontsize=12
    linewidth=1.25
    axes_linewidth=1.35
    tick_linewidth=axes_linewidth*.9
    tick_length=tick_linewidth*5

    @staticmethod
    def read_dat(
        path
    ):
        df_list = []
        if not (isinstance(path, list) or isinstance(path, tuple)):
            path = [path]
        for p in path:
            p = pathlib.Path(p)
            with p.open() as f:
                line_list = []
                i = 0
                for line in f:
                    if line.startswith('#'):
                        i += 1
                        line_list.append(line)
                df = pd.read_csv(p, skiprows=i, sep=r'\s+', engine='python', header=None)
            df_list.append(df)
        return df_list


    @staticmethod
    def plot_r(
        df,
        dim=(3.25,3.25),
        plot_fit=False,
        savefig=None,
        fontsize=fontsize,
        linewidth=linewidth,
        color='#4298B5',
        fig=None,
        ax=None,
        errorbar=None,
        xlim=(0,6),
        ylim=None,
        window=None,
        legend=False,
        legend_loc='lower right',
        legend_fontsize=None,
        dy=0,
        **kwargs
        ):
        if not (fig and ax):
            fig, ax = plt.subplots(layout='constrained')
        
        if plot_fit:
            ax.plot(df[0], df[5]+dy, color=color, zorder=999, linewidth=linewidth, linestyle='--')
            ax.plot(df[0], df[6]+dy, color=color, zorder=998, linewidth=linewidth, linestyle='--')
            if errorbar:
                ax.errorbar(df[0], df[1]+dy, yerr=df[2], fmt='-', color='black')
                ax.errorbar(df[0], df[3]+dy, yerr=df[4], fmt='-', color='black')
            else:
                ax.plot(df[0], df[1]+dy, color='black')
                ax.plot(df[0], df[3]+dy, color='black')
        else:
            if errorbar:
                ax.errorbar(df[0], df[1]+dy, yerr=df[2], fmt='-', color=color)
                ax.errorbar(df[0], df[3]+dy, yerr=df[4], fmt='-', color=color)
            else:
                ax.plot(df[0], df[1]+dy, color=color, linestyle='-')
                ax.plot(df[0], df[3]+dy, color=color, linestyle='-')
                
        if window is not None:
            xy_list = ((window[0]-10,-500), (window[1],-500))
            width_list = (10, 10)
            for i in range(len(xy_list)):
                ax.add_patch(
                    patches.Rectangle(
                        xy_list[i],
                        width_list[i],
                        9999,
                        color='gray',
                        alpha=0.25,
                        zorder=0
                    )
                )
                
        Plot.ax_opts(
            ax,
            xlim=xlim,
            ylim=ylim,
            fontsize=fontsize,
            label_preset='rs_exafs',
            **kwargs
        )
        
        # ax.legend(
        #     # handles=[r"$|\chi(R)|$", r"Re[$\chi(R)$]"], 
        #     # loc=legend_loc,
        #     frameon=False, 
        #     fontsize=fontsize, 
        #     labelspacing=0.075/2, 
        #     borderpad=0, 
        #     handlelength=1, 
        #     handletextpad=0.2
        # )
        
        if legend:
            exp = Line2D([0], [0], label='Data', color='k', linewidth=linewidth)
            handles = [exp]
            if plot_fit:
                fit = Line2D([0], [0], label='Fit', color=color, linestyle='--', markersize=linewidth)
                handles.append(fit)

            if legend_fontsize is None:
                legend_fontsize = fontsize

            ax.legend(handles=handles, 
                loc=legend_loc,
                frameon=False, 
                fontsize=legend_fontsize, 
                labelspacing=0.25/2, 
                borderpad=0, 
                handlelength=1, 
                handletextpad=0.2
)
        
        if dim is not None:
            fig.set_size_inches(*dim)
        if savefig:
            fig.savefig(savefig)
            

    def plot_k(
        df,
        dim=(3.25,3.25),
        plot_fit=False,
        plot_filtered=True,
        plot_window=False,
        savefig=None,
        xlim=(2.5,16),
        ylim=None,
        fontsize=fontsize,
        linewidth=linewidth,
        fit_color='#4298B5',
        fig=None,
        ax=None,
        xcol=5,
        ycol=6,
        kwt=None,
        legend=False,
        legend_loc='lower right',
        legend_fontsize=None,
        **kwargs
        ):
        if fig is None and (ax is None):
            fig, ax = plt.subplots(layout='constrained')
        # df.plot(x=0, y=3, ax=ax)
        if kwt:
            ax.plot(df[xcol], df[ycol]*df[xcol]**kwt, color='gray', linewidth=linewidth)
        else:
            ax.plot(df[xcol], df[ycol], color='gray', linewidth=linewidth)
        
        if plot_filtered:
            ax.plot(df[0], df[1], color='black', linewidth=linewidth)
        if plot_fit:
            ax.plot(df[0], df[3], color=fit_color, zorder=999, linewidth=linewidth, linestyle='--')
        if plot_window:
            ax.plot(df[5], df.iloc[:, -3], color=fit_color, zorder=999, linewidth=linewidth)
        
        # ax.errorbar(df[5], df[6], yerr=df[7], fmt='-', color='black')
        # ax.errorbar(df[0], df[3], yerr=df[4], fmt='+', color='black')
        # df.plot(x=0, y=1, ax=ax)
        # ax.plot(df[0], np.sqrt(df[1]**2+df[6]**2))

        Plot.ax_opts(
            ax,
            xlim=xlim,
            ylim=ylim,
            fontsize=fontsize,
            label_preset='ks_exafs',
            **kwargs
        )
        
        if legend:
            exp = Line2D([0], [0], label='Data', color='gray', linewidth=linewidth)
            expf = Line2D([0], [0], label='Filtered Data', color='k', linewidth=linewidth)
            handles = [exp,expf]
            if plot_fit:
                fit = Line2D([0], [0], label='Fit', color=fit_color, linestyle='--', markersize=linewidth)
                handles.append(fit)

            if legend_fontsize is None:
                legend_fontsize = fontsize

            ax.legend(handles=handles, 
                loc=legend_loc,
                frameon=False, 
                fontsize=legend_fontsize, 
                labelspacing=0.25/2, 
                borderpad=0, 
                handlelength=1, 
                handletextpad=0.2
)
        
        if dim is not None:
            fig.set_size_inches(*dim)
        if savefig:
            fig.savefig(savefig)
            
            
    @classmethod
    def plot_xanes(
        cls,
        df_list,
        dim=(3.25,3.25),
        savefig=None,
        xlim=None,
        ylim=None,
        fontsize=fontsize,
        linewidth=linewidth,
        color=None,
        fig=None,
        ax=None,
        xcol=0,
        ycol=1,
        offset=0,
        text=None,
        legend_fontsize=None,
        **kwargs
    ):
        if fig is None and (ax is None):
            fig, ax = plt.subplots(layout='constrained')
        if color is None:
            color = Plot.sample_colormap(
                'plasma',
                len(df_list)
            )
            
        if isinstance(xcol, int):
            xcol = [xcol for _ in range(len(df_list))]
        if isinstance(ycol, int):
            ycol = [ycol for _ in range(len(df_list))]
            
        i = 0
        for df in df_list:
            ax.plot(df[xcol[i]], df[ycol[i]]+offset*i, color=color[i], linewidth=linewidth)
            i += 1
            
        # text_list = ['Ag foil', 'PAF-1-3S-Ag', r'Ag$_2$S']
        # color_list = [spectrum_color[0], spectrum_color[1], spectrum_color[3]]
        # for i in range(len(text_list)):
        #     ax.text(
        #         25596, 
        #         0.295+0.5+dy*i, 
        #         text_list[i], 
        #         fontsize=fontsize,
        #         horizontalalignment='right',
        #         color=color_list[i]
        #         )

        Plot.ax_opts(
            ax,
            xlim=xlim,
            ylim=ylim,
            fontsize=fontsize,
            label_preset='xanes',
            **kwargs
        )

        if dim is not None:
            fig.set_size_inches(*dim)
        if savefig:
            fig.savefig(savefig)
            
            
class Sp8:
    
    def __init__(
        self,
        dir,
        *args,
        data_list=None,
        E_shift=0,
        is_fluorescence=False,
        import_data_kwargs=None,
        **kwargs
    ):
        if import_data_kwargs is None:
            import_data_kwargs = {}
        if 'E_shift' not in import_data_kwargs.keys():
            import_data_kwargs['E_shift'] = E_shift
        if 'is_fluorescence' not in import_data_kwargs.keys():
            import_data_kwargs['is_fluorescence'] = is_fluorescence
        if data_list is not None:
            self.data_list = data_list
        else:
            self.path_generator(dir, *args, **kwargs)
        self.make_arr_list(**import_data_kwargs)
        # self.arr_list = []
        # for f in self.data_list:
        #     self.arr_list.append(self.import_data(f, **import_data_kwargs))
        
        self.transmission_data = []
        self.fluorescence_channels = []
    
    def path_generator(
        self,
        dir,
        suffix,
        runs,
        scans=None,
        ext='.dat',
        run_format='04',
        scan_format='03',
        exclude_scans=None,
        run_scan_sep='_'
    ):
        '''
        Returns a list of data file paths (pathlib.Path() instances).
        '''
        if exclude_scans is None:
            exclude_scans = ()
        
        self.dir = pathlib.Path(dir)
        # glob everything
        self.data_glob = sorted(self.dir.glob(f"{suffix}*{ext}"))
        # filter runs
        # run ID string formatting
        runs = [format(r, run_format) for r in runs]
        # select only scans belonging to the specified runs
        # self.data_list = [d for d in self.data_list if d.stem[(-int(run_format)-int(scan_format)-len(run_scan_sep)):(-int(scan_format)-len(run_scan_sep))] in runs]
        self.data_list = []
        for d in self.data_glob:
            if d.stem[(-int(run_format)-int(scan_format)-len(run_scan_sep)):(-int(scan_format)-len(run_scan_sep))] in runs:
                self.data_list.append(d)
            elif run_scan_sep not in d.stem and (d.stem[-int(run_format):] in runs):
                self.data_list.append(d)
        # select scans
        if scans is not None:
            if isinstance(scans, int):
                scans = [s for s in range(1, scans+1)]
            # scan ID string formatting
            scans = [format(s, scan_format) for s in scans if s not in exclude_scans]
            self.data_list = [d for d in self.data_list if d.stem[-int(scan_format):] in scans]
        return self.data_list
    
    def import_data(
        self,
        path,
        data_list=None,
        header=None,
        left_str="D=",
        right_str="A",
        E_shift=0,
        ndch=None,
        drop_trailing_raw_cols=None,
        is_fluorescence=False,
        **kwargs
    ):
        '''
        Reads a raw SPring-8 data file and outputs an array of mu vs. E.
        '''
        if data_list is not None:
            self.data_list = data_list
            
        df = pd.read_csv(
            path,
            header=header,
            **kwargs
        )
        
        # get monochromator d-spacing
        str = df[df.iloc[:, 0].str.contains('D=')].iloc[0, 0]
        d_spacing = float(str[str.index(left_str)+len(left_str):str.index(right_str)])
        
        # get number of detector channels
        if ndch is None:
            str = df[df.iloc[:, 0].str.contains('NDCH =')].iloc[0, 0]
            ndch = int(str[-1])
        
        # find start of data
        skiprows = df[df.iloc[:, 0].str.contains('Offset')].index.values[0] + 1
        
        # slice and dice
        raw_arr = df.iloc[skiprows:, 0].str.split().apply(pd.to_numeric).apply(pd.Series).to_numpy()
        if drop_trailing_raw_cols is None:
            if ndch > 3:
                drop_trailing_raw_cols = ndch
            else:
                drop_trailing_raw_cols = 0
        
        if drop_trailing_raw_cols > 0:
            raw_arr = raw_arr[:, 0:-drop_trailing_raw_cols]
        
        energy = self.energy(raw_arr[:, 1], d_spacing) + E_shift
        
        transmission_data = np.empty((len(raw_arr), 2))
        transmission_data[:, 0] = energy
        transmission_data[:, 1] = -np.log(raw_arr[:, -1]/raw_arr[:, -2])
        transmission_data = sort_by_column(transmission_data)
        
        if ndch > 3 and is_fluorescence:
            fluorescence_channels = np.empty((len(raw_arr), ndch))
            fluorescence_channels[:, 0] = energy
            for i in range(ndch-2):
                fluorescence_channels[:, i+1] = raw_arr[:, i+3] / raw_arr[:, -2]
            fluorescence_channels[:, -1] = np.average(fluorescence_channels[:, 1:ndch-1], axis=1)
            fluorescence_channels = sort_by_column(fluorescence_channels)
            return np.column_stack((fluorescence_channels[:, 0], fluorescence_channels[:, -1]))
        else:
            return transmission_data
            
    def make_arr_list(
        self,
        arr_list=None,
        data_list=None,
        skipnan=True,
        **kwargs
    ):
        if data_list is not None:
            self.data_list = data_list
        if arr_list is None:
            self.arr_list = []
        elif arr_list == 'append':
            pass
        else:
            self.arr_list = arr_list
        for f in self.data_list:
            arr = self.import_data(f, **kwargs)
            if skipnan and (np.isnan(arr).any()):
                warnings.warn(f'{f} returned NaN values and was dropped.')
            else:
                self.arr_list.append(arr)
        
    def dump_ascii(
        self,
        output_file,
        save_norm=False,
        **kwargs
    ):
        if save_norm:
            if 'label' not in kwargs.keys():
                kwargs['label'] = 'energy mu norm'
            lio.write_ascii(
                output_file, 
                self.group.energy, 
                self.group.mu, 
                self.group.norm, 
                **kwargs
            )
        else:
            if 'label' not in kwargs.keys():
                kwargs['label'] = 'energy mu'
            lio.write_ascii(
                output_file, 
                self.group.energy, 
                self.group.mu, 
                **kwargs
            )
        
    def shift_energy(
        self,
        E_shift=None,
        E0_ref=None,
        E0_act=None
    ):
        '''
        Shifts all individual scans as well as the averaged mu vs E array and Larch group, if found.
        E_shift takes priority if it is specified.
        Otherwise, E_shift is calculated from the difference of E0_ref and E0_act.
        By default, E0_act is taken from self.group.e0 found by Larch.
        '''
        if E0_act is None:
            E0_act = self.group.e0
        if E_shift is None:
            E_shift = E0_ref - E0_act
        for a in self.arr_list:
            a[:, 0] += E_shift
        # try:
        #     self.arr_avg[:, 0] += E_shift
        # except:
        #     warnings.warn('arr_avg not found for E0 shift')
        #     pass
        try:
            self.group.energy += E_shift
        except:
            warnings.warn('Larch group not found for E0 shift')
            pass
        try:
            self.group.e0 += E_shift
        except:
            pass

    
    
    def interpolate_and_average(
        self,
        bounds=None, 
        step=0.1,
        grid_points=None,
        # s=0.01,
        kind='cubic',
        tiny=1e-6,
        **kwargs
    ):
        # kwargs['kind'] = kind
        # if grid_points is None:
        #     # attempt to use min and max E from first array
        #     if bounds is None:
        #         bounds = [
        #             np.ceil(min(self.arr_list[0][:, 0]/step))*step,
        #             np.floor(max(self.arr_list[0][:, 0]/step))*step
        #         ]
        #     grid_points = np.arange(bounds[0], bounds[1], step)
        # self.mu_interp = np.empty((len(grid_points), len(self.arr_list)))
        # self.arr_avg = np.empty((len(grid_points), 2))
        # for i, arr in enumerate(self.arr_list):
        #     self.mu_interp[:, i] = interp1d(
        #         remove_dups(arr[:, 0], tiny=tiny),
        #         arr[:, 1],
        #         grid_points,
        #         **kwargs
        #     )
        #     # self.mu_interp[:, i] = self.interpolate_and_resample(
        #     # arr, bounds=bounds, step=step, grid_points=grid_points, s=s
        #     # )[:, 1]
        # self.arr_avg[:, 0] = grid_points
        # self.arr_avg[:, 1] = np.average(self.mu_interp, axis=1)
        # self.arr_avg = self.arr_avg[~np.isnan(self.arr_avg).any(axis=1)]
        
        
        grouplist = []
        for i, arr in enumerate(self.arr_list):
            group = Group()
            group.energy = arr[:, 0]
            group.mu = arr[:, 1]
            grouplist.append(group)
            
        merged_group = lio.merge_groups(grouplist, **kwargs)
        self.arr_avg = np.empty((len(merged_group.energy), 2))
        self.arr_avg[:, 0] = merged_group.energy
        self.arr_avg[:, 1] = merged_group.mu
        
        self.E_max = self.arr_avg[self.arr_avg[:, 1].argmax(), 0]
        
        
    def check_average(
        self,
        fig=None,
        ax=None,
        **kwargs
    ):
        if fig is None and (ax is None):
            fig, ax = plt.subplots(**kwargs)
        for a in self.arr_list:
            ax.plot(a[:, 0], a[:, 1])
        ax.plot(self.arr_avg[:, 0], self.arr_avg[:, 1], 'k-')
        try:
            ax.plot(self.group.energy, self.group.mu, 'k-')
        except:
            pass
        return fig, ax
    
    def make_group(
        self,
        pre_edge=True,
        group_kwargs=None,
        **kwargs
    ):
        '''
        Create a Larch group from the averaged mu vs. E array.
        Optionally, get E0 and do pre-edge subtraction.
        '''
        if group_kwargs is None:
            group_kwargs = {}
        self.group = Group(**group_kwargs)
        self.group.energy = self.arr_avg[:, 0]
        self.group.mu = self.arr_avg[:, 1]
        if pre_edge:
            lx.pre_edge(
                self.group,
                **kwargs
            )
    
    @classmethod
    def energy(cls, theta, d, n=1):
        '''
        Calculates energy from monochromator orientation and angle using Bragg's law.
        '''
        return 12398 / cls.bragg(theta, d, n)
    
    @staticmethod
    def bragg(theta, d, n=1):
        return n * 2 * d * np.sin(np.radians(theta))
    
    @staticmethod
    def interpolate_and_resample(
        arr, 
        bounds=None, 
        step=None,
        grid_points=None,
        s=0.001
    ):
        """
        Use splines to reevaluate a DataFrame on a specified grid for a given column.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to reevaluate.
        column (str): The column name to base the reevaluation on.
        grid_points (array-like): The grid points to reevaluate the DataFrame on.
        
        Returns:
        pd.DataFrame: The reevaluated DataFrame.
        """
        if grid_points is None:
            grid_points = np.arange(bounds[0], bounds[1], step)
        
        arr_out = np.empty((len(grid_points), 2))
        # print(arr[:, 0])
        spline = UnivariateSpline(arr[:, 0], arr[:, 1], s=s)
        arr_out[:, 0] = grid_points
        arr_out[:, 1] = spline(grid_points)
        return arr_out
    
    
class Larch:
    
    def __init__(
        self,
        groups: dict,
    ):
        """Initializes self.groups.

        Args:
            groups (dict): Dictionary of Larch groups.
        """
        self.groups = groups
        self.feffit_run_index: int = 0
        self.feffit_outputs: dict = {self.feffit_run_index: None}
    
    def merge_groups(self, *args, keys=None, merge_group_key: str | None = None, overwrite: bool = True, drop_merged=False, **kwargs):
        """Wrapper for `lio.merge_groups` that accepts keys from `self.groups` and
        stores the resulting merged group back into `self.groups`.

        Parameters
        ----------
        keys : str or iterable
            A single key or an iterable of keys referring to entries in
            `self.groups`.
        *args, **kwargs
            Forwarded to `lio.merge_groups`.
        merge_group_key : str | None
            Optional key name to use when storing the merged group in
            `self.groups`. If `None`, a key is auto-generated from the
            merged input keys (see implementation).

        Returns
        -------
        group
            The merged Larch group returned by `lio.merge_groups`.
        """
        # Normalize keys to a list of strings
        if keys is None:
            keys = list(self.groups.keys())
        elif isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, (list, tuple, set)):
            try:
                keys = list(keys)
            except TypeError:
                keys = [keys]

        # Lookup group objects
        groups = [self.groups[k] for k in keys]

        # Call lio.merge_groups and get the resulting group
        print(groups)
        merged = lio.merge_groups(groups, *args, **kwargs)

        # Decide on a key for the merged group and store it
        if merge_group_key is None:
            # attempt to compute a sensible stem from the input keys
            try:
                cp = os.path.commonprefix(keys)
            except Exception:
                cp = ''

            stem = cp.rstrip('_')
            if not stem:
                # fallback: use prefix of first key up to last '_'
                first = keys[0]
                if '_' in first:
                    stem = first.rsplit('_', 1)[0]
                else:
                    stem = first

            merge_group_key = f"{stem}_merged_{len(keys)}"

            # ensure uniqueness in self.groups
            base = merge_group_key
            i = 1
            while merge_group_key in self.groups:
                i += 1
                merge_group_key = f"{base}_{i}"

        # If user provided a key that already exists, warn and respect overwrite
        if merge_group_key in self.groups:
            warnings.warn(f"Group '{merge_group_key}' already exists.")
            if not overwrite:
                return self.groups[merge_group_key]

        # store (or overwrite) merged group
        self.groups[merge_group_key] = merged
        
        if drop_merged:
            drop_keys = [k for k in self.groups.keys() if k is not merge_group_key]
            self.drop_groups(drop_keys)

        return merged

    def drop_groups(self, keys, missing: str = 'ignore'):
        """Drop one or more groups from ``self.groups`` by key.

        Parameters
        ----------
        keys : str or iterable
            Single key or iterable of keys to remove from ``self.groups``.
        missing : {'ignore', 'warn', 'raise'}, optional
            Behavior when a requested key is not present. Defaults to 'ignore'.

        Returns
        -------
        list
            List of keys that were removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, (list, tuple, set)):
            try:
                keys = list(keys)
            except TypeError:
                keys = [keys]

        removed = []
        for k in keys:
            if k in self.groups:
                del self.groups[k]
                removed.append(k)
            else:
                if missing == 'warn':
                    warnings.warn(f"Group '{k}' not found; skipping drop.")
                elif missing == 'raise':
                    raise KeyError(f"Group '{k}' not found in Larch.groups")
                # if 'ignore', do nothing

        return removed
    
    def plot_ekr_single(
        self,
        key,
        fig=None,
        axs=None,
       **kwargs,
    ):
        if (axs is None) or (fig is None):
            fig, axs = plt.subplots(nrows=1, ncols=3, layout='constrained')
            
        group = self.groups[key]
        # if no explicit label provided, use the key as the label
        if 'label' not in kwargs:
            kwargs['label'] = key
        fig, axs = plot_ekr(
            group,
            fig=fig,
            axs=axs,
            **kwargs,
        )
        
        return fig, axs
    
    def autobk_xftf_single(
        self,
        key,
        **kwargs,
    ):
        group = self.groups[key]
        autobk_xftf(
            key,
            self.groups[key],
            **kwargs,
        )
    
    def plot_ekr(
        self,
        keys=None,
        autobk_xftf=True,
        autobk_kwargs=None,
        xftf_kwargs=None,
        **kwargs,
    ):
        if keys is None:
            keys = self.groups.keys()
        
        if autobk_kwargs is not None:
            autobk_kwargs = self.check_nested_dictionaries(autobk_kwargs)
        else:
            autobk_kwargs = dict(zip(self.groups.keys(), [None for _ in range(len(self.groups))]))
        
        if xftf_kwargs is not None:
            xftf_kwargs = self.check_nested_dictionaries(xftf_kwargs)
        else:
            xftf_kwargs = dict(zip(self.groups.keys(), [None for _ in range(len(self.groups))]))
            
        self.fig_ax_outputs = {}
        for k in keys:
            if autobk_xftf:
                self.autobk_xftf_single(
                    k,
                    autobk_kwargs=autobk_kwargs[k],
                    xftf_kwargs=xftf_kwargs[k],
                )
            
            fig, axs = self.plot_ekr_single(
                k,
                **kwargs,
            )
            self.fig_ax_outputs[k] = (fig, axs)
        
        return self.fig_ax_outputs
    
    def plot_k_fitted(
        self,
        dset,
        key: str,
        k_weight: int = 3,
        plot_data_kwargs: dict | None = None,
        plot_model_kwargs: dict | None = None,
        ax_opts_kwargs: dict | None = None,
        legend: bool = True,
        fig=None,
        ax=None,
    ):
        """Plot k-space fitted data and model.
        
        Parameters
        ----------
        dset : FeffitDataset
            The dataset containing data and model.
        key : str
            Key name for labeling.
        k_weight : int, optional
            k-weighting power, by default 3.
        plot_data_kwargs : dict, optional
            Kwargs for plotting data.
        plot_model_kwargs : dict, optional
            Kwargs for plotting model.
        ax_opts_kwargs : dict, optional
            Kwargs for axis options.
        legend : bool, optional
            Whether to show legend, by default True.
        fig : Figure, optional
            Matplotlib figure.
        ax : Axes, optional
            Matplotlib axes.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        DEFAULT_DATA_COLOR = 'k'
        DEFAULT_DATA_LINESTYLE = '-'
        DEFAULT_MODEL_COLOR = '#4298B5'
        DEFAULT_MODEL_LINESTYLE = '--'
        DEFAULT_FONTSIZE = 12
        
        if plot_data_kwargs is None:
            plot_data_kwargs = {}
        if plot_model_kwargs is None:
            plot_model_kwargs = {}
        if ax_opts_kwargs is None:
            ax_opts_kwargs = {}
            
        if 'color' not in plot_data_kwargs:
            plot_data_kwargs['color'] = DEFAULT_DATA_COLOR
        if 'color' not in plot_model_kwargs:
            plot_model_kwargs['color'] = DEFAULT_MODEL_COLOR
        if 'linestyle' not in plot_data_kwargs:
            plot_data_kwargs['linestyle'] = DEFAULT_DATA_LINESTYLE
        if 'linestyle' not in plot_model_kwargs:
            plot_model_kwargs['linestyle'] = DEFAULT_MODEL_LINESTYLE
        if 'fontsize' not in ax_opts_kwargs:
            ax_opts_kwargs['fontsize'] = DEFAULT_FONTSIZE
            
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            
        ax.plot(dset.data.k, dset.data.chi*dset.data.k**k_weight, **plot_data_kwargs)
        ax.plot(dset.model.k, dset.model.chi*dset.data.k**k_weight, **plot_model_kwargs)
        
        Plot.ax_opts(ax, **ax_opts_kwargs)
        
        if legend:
            custom_lines = [
                Line2D([0], [0], color=plot_data_kwargs['color'], linestyle=plot_data_kwargs['linestyle'], label='Data'),
                Line2D([0], [0], color=plot_model_kwargs['color'], linestyle=plot_model_kwargs['linestyle'], label='Fit')
            ]
            legend_kwargs = {
                'frameon': False,
                'fontsize': ax_opts_kwargs['fontsize'],
                'labelspacing': 0.25,
                'handlelength': 1.2
            }
            ax.legend(handles=custom_lines, loc='lower right', **legend_kwargs)
            
        return fig, ax
    
    def plot_r_fitted(
        self,
        dset,
        key: str,
        k_weight: int = 3,
        plot_data_kwargs: dict | None = None,
        plot_model_kwargs: dict | None = None,
        ax_opts_kwargs: dict | None = None,
        plot_fit_window: bool = True,
        plot_model: bool = True,
        legend: bool = True,
        plot_text: bool = True,
        fig=None,
        ax=None,
    ):
        """Plot r-space fitted data and model.
        
        Parameters
        ----------
        dset : FeffitDataset
            The dataset containing data and model.
        key : str
            Key name for labeling.
        k_weight : int, optional
            k-weighting power (not used in r-space but kept for consistency), by default 3.
        plot_data_kwargs : dict, optional
            Kwargs for plotting data.
        plot_model_kwargs : dict, optional
            Kwargs for plotting model.
        ax_opts_kwargs : dict, optional
            Kwargs for axis options.
        plot_fit_window : bool, optional
            Whether to plot fit window, by default True.
        legend : bool, optional
            Whether to show legend, by default True.
        plot_text : bool, optional
            Whether to show key text, by default True.
        fig : Figure, optional
            Matplotlib figure.
        ax : Axes, optional
            Matplotlib axes.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        DEFAULT_DATA_COLOR = 'k'
        DEFAULT_DATA_LINESTYLE = '-'
        DEFAULT_MODEL_COLOR = '#4298B5'
        DEFAULT_MODEL_LINESTYLE = '--'
        DEFAULT_FONTSIZE = 12
        
        if plot_data_kwargs is None:
            plot_data_kwargs = {}
        if plot_model_kwargs is None:
            plot_model_kwargs = {}
        if ax_opts_kwargs is None:
            ax_opts_kwargs = {}
            
        if 'color' not in plot_data_kwargs:
            plot_data_kwargs['color'] = DEFAULT_DATA_COLOR
        if 'color' not in plot_model_kwargs:
            plot_model_kwargs['color'] = DEFAULT_MODEL_COLOR
        if 'linestyle' not in plot_data_kwargs:
            plot_data_kwargs['linestyle'] = DEFAULT_DATA_LINESTYLE
        if 'linestyle' not in plot_model_kwargs:
            plot_model_kwargs['linestyle'] = DEFAULT_MODEL_LINESTYLE
        if 'fontsize' not in ax_opts_kwargs:
            ax_opts_kwargs['fontsize'] = DEFAULT_FONTSIZE
            
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            
        rmin = dset.transform.rmin
        rmax = dset.transform.rmax
        
        ax.plot(dset.data.r, dset.data.chir_mag, label="Data", **plot_data_kwargs)
        ax.plot(dset.data.r, dset.data.chir_re, **plot_data_kwargs)
        if plot_model:
            ax.plot(dset.model.r, dset.model.chir_mag, **plot_model_kwargs)
            ax.plot(dset.model.r, dset.model.chir_re, **plot_model_kwargs)
        
        if plot_text:
            ax.text(
                0.95,
                0.95,
                key,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=ax_opts_kwargs['fontsize'],
            )
            
        if plot_fit_window:
            ax.add_patch(
                patches.Rectangle(
                    (rmin, -5000),
                    rmax - rmin,
                    10000,
                    color=plot_model_kwargs['color'],
                    alpha=0.1,
                    zorder=0
                )
            )
            
        Plot.ax_opts(ax, **ax_opts_kwargs)
        
        if legend:
            custom_lines = [
                Line2D([0], [0], color=plot_data_kwargs['color'], linestyle=plot_data_kwargs['linestyle'], label='Data'),
                Line2D([0], [0], color=plot_model_kwargs['color'], linestyle=plot_model_kwargs['linestyle'], label='Fit')
            ]
            legend_kwargs = {
                'frameon': False,
                'fontsize': ax_opts_kwargs['fontsize'],
                'labelspacing': 0.25,
                'handlelength': 1.2
            }
            ax.legend(handles=custom_lines, loc='lower right', **legend_kwargs)
            
        return fig, ax
    
    def plot_kr_fitted(
        self,
        keys=None,
        feffit_run_outputs: dict | None = None,
        feffit_run_index: int | None = None,
        k_weight: int = 3,
        k_plot_data_kwargs: dict | None = None,
        r_plot_data_kwargs: dict | None = None,
        k_plot_model_kwargs: dict | None = None,
        r_plot_model_kwargs: dict | None = None,
        k_ax_opts_kwargs: dict | None = None,
        r_ax_opts_kwargs: dict | None = None,
        plot_fit_window: bool = True,
        fig_dimensions_inches: list | None = [6.5, 3.25],
        legend: bool = True,
        file_name_prefix: str | None = None,
        save_directory: pathlib.Path | None = None,
        plot_text: bool = True,
    ):  
        """Plot k-space and r-space fitted data and model in a two-panel figure.
        
        This method creates two-panel plots by calling plot_k_fitted() and plot_r_fitted().
        """
        ## if no run index is specified, will use the last set one
        if feffit_run_index is None:
            feffit_run_index = self.feffit_run_index
        
        ## if no dictionary of FEFFIT outputs is given, will pull from cached runs
        if feffit_run_outputs is None:
            feffit_run_outputs = self.feffit_outputs[feffit_run_index]
            
        ## handle output from feffit_multi_aligned
        if isinstance(feffit_run_outputs, list):
            feffit_run_outputs = feffit_run_outputs[0]
            
        if keys is None:
            keys = feffit_run_outputs.keys()
            
        fig_ax_outputs = {}
        for k in keys:
            v = feffit_run_outputs[k]
            if isinstance(v, list) or isinstance(v, tuple):
                dset, _ = v
            else:
                dset = v
            
            # Create two-panel figure
            fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained', sharex='col', sharey='col')
            
            # Plot k-space on left panel
            self.plot_k_fitted(
                dset=dset,
                key=k,
                k_weight=k_weight,
                plot_data_kwargs=k_plot_data_kwargs,
                plot_model_kwargs=k_plot_model_kwargs,
                ax_opts_kwargs=k_ax_opts_kwargs,
                legend=legend,
                fig=fig,
                ax=axs[0],
            )
            
            # Plot r-space on right panel
            self.plot_r_fitted(
                dset=dset,
                key=k,
                k_weight=k_weight,
                plot_data_kwargs=r_plot_data_kwargs,
                plot_model_kwargs=r_plot_model_kwargs,
                ax_opts_kwargs=r_ax_opts_kwargs,
                plot_fit_window=plot_fit_window,
                legend=legend,
                plot_text=plot_text,
                fig=fig,
                ax=axs[1],
            )
            
            if fig_dimensions_inches is not None:
                fig.set_size_inches(*fig_dimensions_inches)
                
            if file_name_prefix is not None:
                file_name = str(file_name_prefix) + '_'
            else:
                file_name = ''
                
            if save_directory is not None:
                file_name += f"{k}.svg"
                fig.savefig(pathlib.Path(save_directory) / file_name)
                
            fig_ax_outputs[k] = (fig, axs)
            
        return fig_ax_outputs
    
    def feffit_single(
        self,
        key,
        feff_paths,
        parameter_group,
        autobk_kwargs: dict | None = None,
        xftf_kwargs: dict | None = None,
        method: str = 'leastsq',
        group=None,
        **kwargs,
    ):
        if group is None:
            group = self.groups[key]
            
        if autobk_kwargs is not None:
            lx.autobk(group.energy, group.norm, group=group, **autobk_kwargs)
            
        transform = lx.feffit_transform(**xftf_kwargs)
        feffit_dataset = lx.feffit_dataset(data=group, pathlist=feff_paths, transform=transform)
        feffit_output = lx.feffit(parameter_group, [feffit_dataset], method=method, **kwargs)
        return [feffit_dataset, feffit_output]
    
    def feffit(
        self,
        feff_paths,
        parameter_group,
        autobk_kwargs: dict | None = None,
        xftf_kwargs: dict | None = None,
        keys: list[str] | None = None,
        method: str = 'leastsq',
        file_name_prefix: str | None = None,
        save_directory: pathlib.Path | None = None,
        feffit_run_index: int | None = None,
        **kwargs,
    ):
        """
        Fits the FEFF paths specified by feff_paths, parameterized by parameter_groups.
        Outputs a dictionary of [feffit_dataset, feffit_output] for each group that was (individually) fitted.
        
        """
        if keys is None:
            keys = self.groups.keys()
            
        if xftf_kwargs is not None:
            xftf_kwargs = self.check_nested_dictionaries(xftf_kwargs)
        else:
            xftf_kwargs = dict(zip(self.groups.keys(), [None for _ in range(len(self.groups))]))

        if autobk_kwargs is not None:
            autobk_kwargs = self.check_nested_dictionaries(autobk_kwargs)
        else:
            autobk_kwargs = dict(zip(self.groups.keys(), [None for _ in range(len(self.groups))]))
            
        feffit_run_outputs = {}
        ## auto-assign run index by incrementing to prevent accidental overwrite
        if feffit_run_index is None:
            self.feffit_run_index = max(self.feffit_outputs) + 1
        else:
            self.feffit_run_index = feffit_run_index
            
        for k in keys:
            feffit_run_outputs[k] = self.feffit_single(
                k,
                feff_paths,
                parameter_group,
                autobk_kwargs=autobk_kwargs[k],
                xftf_kwargs=xftf_kwargs[k],
                method=method,
                **kwargs,
            )
            if file_name_prefix is not None:
                file_name = str(file_name_prefix) + '_'
            else:
                file_name = ''
                
            if save_directory is not None:
                file_name += f"{k}_run{self.feffit_run_index}.txt"
                with open(pathlib.Path(save_directory) / file_name, 'w') as f:
                    f.write(lx.feffit_report(feffit_run_outputs[k][1]))
            
        self.feffit_outputs[self.feffit_run_index] = feffit_run_outputs
        return feffit_run_outputs
    
    def feffit_multi_aligned(
        self,
        std_dir: pathlib.Path,
        path_dict: dict,
        linked_params: list | tuple,
        feffit_run_index: int | None = None,
        **kwargs,
    ):
        if 'keys' not in kwargs:
            kwargs['keys'] = None
            
        ## auto-assign run index by incrementing to prevent accidental overwrite
        if feffit_run_index is None:
            self.feffit_run_index = max(self.feffit_outputs) + 1
        else:
            self.feffit_run_index = feffit_run_index
            
        feffit_datasets, feffit_output = feffit_multi_aligned(
            self.groups,
            std_dir,
            path_dict,
            linked_params,
            **kwargs,
        )
        
        self.feffit_outputs[self.feffit_run_index] = [feffit_datasets, feffit_output]
    
    def iterative_background_fit(
        self,
        key: str,
        rbkg_list: list,
        rmin_list: list,
        rmax_list: list,
        initial_autobk_kwargs: dict,
        initial_xftf_kwargs: dict,
        feff_paths,
        parameter_group,
        kmin_list: list | None = None,
        kmax_list: list | None = None,
        initial_k_std = None,
        initial_chi_std = None,
        n_iter: int | None = None,
        method: str = 'leastsq',
        file_name_prefix: str | None = None,
        save_directory: pathlib.Path | None = None,
    ):
        if n_iter is None:
            n_iter = len(rbkg_list)
            
        autobk_kwargs = copy.copy(initial_autobk_kwargs)
        xftf_kwargs = copy.copy(initial_xftf_kwargs)
        k_std = initial_k_std
        chi_std = initial_chi_std
        
        group = copy.copy(self.groups[key])
        pars = copy.copy(parameter_group)
            
        feffit_run_outputs = {}
        iterated_groups = []
        for j in range(n_iter):
            autobk_kwargs['rbkg'] = rbkg_list[j]
            autobk_kwargs['k_std'] = k_std
            autobk_kwargs['chi_std'] = chi_std
            xftf_kwargs['rmin'] = rmin_list[j]
            xftf_kwargs['rmax'] = rmax_list[j]
            
            if kmin_list is not None:
                xftf_kwargs['kmin'] = kmin_list[j]
                
            if kmax_list is not None:
                xftf_kwargs['kmax'] = kmax_list[j]
                
            
            feffit_run_outputs[j] = self.feffit_single(
                "",
                feff_paths,
                pars,
                autobk_kwargs=autobk_kwargs,
                xftf_kwargs=xftf_kwargs,
                method=method,
                group=group,
            )
            
            iterated_groups.append(copy.copy(group))
            
            k_std = feffit_run_outputs[j][0].model.k
            chi_std = feffit_run_outputs[j][0].model.chi
            pars = feffit_run_outputs[j][1].params
            
            if file_name_prefix is not None:
                file_name = str(file_name_prefix) + '_'
            else:
                file_name = ''
                
            if save_directory is not None:
                file_name += f"{key}_iteration{j}.txt"
                with open(save_directory / file_name, 'w') as f:
                    f.write(lx.feffit_report(feffit_run_outputs[j][1]))
            
        return feffit_run_outputs, group, xftf_kwargs, iterated_groups
    
    def check_nested_dictionaries(
        self,
        dictionary: dict,
    ):
        if not all([isinstance(v, dict) for v in dictionary.values()]):
            dictionary = dict(zip(self.groups.keys(), [dictionary for _ in range(len(self.groups))]))
        
        return dictionary
    
    
def read_ascii_append(f):
    group = lio.read_ascii(
        f,
        labels='energy mu norm'
    )
    return f.stem, group

def read_files_to_larch(files_dict: dict, groups: dict | None = None, **read_ascii_kwargs):
    if groups is None:
        groups = {}
    if 'labels' not in read_ascii_kwargs:
        read_ascii_kwargs['labels'] = 'energy mu norm'
    for k, f in files_dict.items():
        groups[k] = lio.read_ascii(
            f,
            **read_ascii_kwargs,
        )
    return Larch(groups)

def zipper(out):
    u = []
    v = []
    for o in out:
        u.append(o[0])
        v.append(o[1])
    return dict(zip(u, v))

def plot_ekr(
    group,
    axs=None,
    fig=None,
    dxlim=(-30, 150),
    xlim=None,
    label: str | None = None,
    plot_window: bool = True,
    ):
    if (axs is None) or (fig is None):
        fig, axs = plt.subplots(nrows=1, ncols=3, layout='constrained')
    axs[0].plot(group.energy, group.norm)
    axs[0].plot(group.energy, group.bkg)
    if xlim is None:
        xlim = (group.e0 + dxlim[0], group.e0 + dxlim[1])
    axs[0].set_xlim(xlim)
    axs[1].plot(group.k, group.k**3*group.chi)
    axs[2].plot(group.r, group.chir_re)
    axs[2].plot(group.r, group.chir_mag)
    axs[2].vlines(group.rbkg, -999, 999, linestyle='--', color='gray')
    axs[2].set_xlim(0, 6)
    abs_ylim = max(group.chir_mag) * 1.2
    axs[2].set_ylim(-abs_ylim, abs_ylim)
    if plot_window:
        axs[1].plot(group.k, group.kwin * max(group.k**3*group.chi) * 1.1)
    # optional label in bottom-right of the first axis (axis-relative coords)
    if label is not None:
        axs[0].text(
            0.95,
            0.05,
            label,
            transform=axs[0].transAxes,
            ha='right',
            va='bottom',
        )
    fig.set_size_inches(9, 3)
    return fig, axs

def autobk_xftf(
    k,
    group,
    autobk_kwargs=None,
    xftf_kwargs=None,
):
    if autobk_kwargs is None:
        autobk_kwargs = {}
    if xftf_kwargs is None:
        xftf_kwargs = {}
    lx.autobk(group.energy, group.norm, group=group, **autobk_kwargs)
    lx.xftf(group.k, group.chi, group=group, **xftf_kwargs)
    return k, group

def generate_independent_path_parameters(
    path_dict: dict,
    std_dir: pathlib.Path,
    s02: float = 1.0,
    sig_initial: float = 0.005,
    sig_kwargs: dict = {'min': 0.0001, 'max': 0.05, 'vary': True},
    dr_initial: float = 0.0,
    dr_kwargs: dict = {'min': -0.35, 'max': 0.35, 'vary': True},
    n_kwargs: dict = {'vary': False},
    de0_kwargs: dict = {'value': 0.0, 'min': -20.0, 'max': 20.0, 'vary': True},
    override_degen=True,
):
    parameters = {}
    feff_paths = []

    for i, (k, n) in enumerate(path_dict.items()):
        sigma2 = f'sig_{i}'
        deltar = f'dr_{i}'
            
        parameters[f'sig_{i}'] = param(sig_initial, **sig_kwargs)
        parameters[f'dr_{i}'] = param(dr_initial, **dr_kwargs)
        parameters[f'n_{i}'] = param(n, **n_kwargs)
        if override_degen:
            feffpath = lx.feffpath(
                std_dir / k,
                s02=f's02*n_{i}',
                e0='de0',
                deltar=deltar,
                sigma2=sigma2,
                degen=1
            )
        else:
            feffpath = lx.feffpath(
                std_dir / k,
                s02=f's02*n_{i}',
                e0='de0',
                deltar=deltar,
                sigma2=sigma2,
            )
            
        feff_paths.append(feffpath)
        
    parameter_group = param_group(
        s02     = param(s02, vary=False, min=0.5, max=1.1),
        de0     = param(**de0_kwargs),
        **parameters
    )
    
    return feff_paths, parameter_group

def generate_pathlist(
    std_dir: pathlib.Path,
    path_dict: dict,
    linked_params: list | tuple,
    param_suffix: str,
    parameter_dict: dict,
    sig_initial: float = 0.005,
    sig_kwargs: dict = {'min': 0.0001, 'max': 0.05, 'vary': True},
    dr_initial: float = 0.0,
    dr_kwargs: dict = {'min': -0.35, 'max': 0.15, 'vary': True},
    n_kwargs: dict = {'vary': False},
    c3_initial: float = 0.0001,
    c3_kwargs: dict | None = None,
):
    LINKABLE_PARAMS = ("n", "de0", "dr", "sig", "c3")
    PARAM_NAMES_TO_FEFFPATH_KWARGS = {
        "n": "s02",
        "de0": "e0",
        "dr": "deltar",
        "sig": "sigma2",
        "c3": "third",
    }
    
    if c3_kwargs is None:
        c3_kwargs = {"vary": False}

    pathlist = []
    for i, (path_name, n) in enumerate(path_dict.items()):
        ## default names for linked parameters for each path (i)
        feffpath_kwargs = {
            "s02": f"s02*n_{i}",
            "e0": f"de0",
            "deltar": f"dr_{i}",
            "sigma2": f"sig_{i}",
            "third": f"c3_{i}",
            "degen": 1,
        }
        ## for independent paths, append a group-specific suffix
        for p in LINKABLE_PARAMS:
            if p not in linked_params:
                feffpath_kwargs[PARAM_NAMES_TO_FEFFPATH_KWARGS[p]] += f"_{param_suffix}"

        parameter_dict[feffpath_kwargs["e0"]] = param(0.0, min=-20.0, max=20.0, vary=True)
        parameter_dict[feffpath_kwargs["sigma2"]] = param(sig_initial, **sig_kwargs)
        parameter_dict[feffpath_kwargs["deltar"]] = param(dr_initial, **dr_kwargs)
        parameter_dict[feffpath_kwargs["s02"][4:]] = param(n, **n_kwargs)
        parameter_dict[feffpath_kwargs["third"]] = param(c3_initial, **c3_kwargs)

        feffpath = lx.feffpath(
            std_dir / path_name,
            **feffpath_kwargs,
        )
        pathlist.append(feffpath)

    return pathlist, parameter_dict

def feffit_multi_aligned(
    groups: dict,
    std_dir: pathlib.Path,
    path_dict: dict,
    linked_params: list | tuple,
    keys=None,
    param_suffixes: dict | None = None,
    method='leastsq',
    xftf_kwargs: dict | None = None,
    sig_initial: float = 0.005,
    sig_kwargs: dict = {'min': 0.0001, 'max': 0.05, 'vary': True},
    dr_initial: float = 0.0,
    dr_kwargs: dict = {'min': -0.35, 'max': 0.15, 'vary': True},
    n_kwargs: dict = {'vary': False},
    c3_initial: float = 0.0001,
    c3_kwargs: dict | None = None,
    s02: float = 1.0,
    save_file: pathlib.Path | None = None,
    pathlist_dict: dict | None = None,
    parameter_dict: dict | None = None,
):
    """
    linked params: de0, n, dr, sig, c3
    """    
    if keys is None:
        keys = groups.keys()
        
    if param_suffixes is None:
        param_suffixes = dict(zip(keys, keys))
        
    if xftf_kwargs is None:
        xftf_kwargs = {}
        
    if c3_kwargs is None:
        c3_kwargs = {"vary": False}
        
    trans = lx.feffit_transform(**xftf_kwargs)

    if parameter_dict is None:
        parameter_dict: dict = {}
        
    feffit_datasets: dict = {}
    
    ## iterate over selected groups
    for key in keys:
        if pathlist_dict is not None and key in pathlist_dict:
            pathlist = pathlist_dict[key]
        else:
            pathlist, parameter_sub_dict = generate_pathlist(
                path_dict=path_dict,
                std_dir=std_dir,
                linked_params=linked_params,
                param_suffix=param_suffixes[key],
                parameter_dict=parameter_dict,
                sig_initial=sig_initial,
                sig_kwargs=sig_kwargs,
                dr_initial=dr_initial,
                dr_kwargs=dr_kwargs,
                n_kwargs=n_kwargs,
                c3_initial=c3_initial,
                c3_kwargs=c3_kwargs,
            )
            parameter_dict.update(parameter_sub_dict)
            
        feffit_datasets[key] = lx.feffit_dataset(data=groups[key], pathlist=pathlist, transform=trans)

    parameter_group = param_group(
        s02=param(s02, vary=False, min=0.5, max=1.25),
        **parameter_dict
    )

    results = lx.feffit(parameter_group, feffit_datasets.values(), method=method)

    if save_file is not None:
        with open(save_file, 'w') as f:
            f.write(lx.feffit_report(results))

    return feffit_datasets, results

def hamilton_f(
    null_r: float,
    alternative_r: float,
    null_parameters: float,
    alternative_parameters: float,
    n_independent: float,
    b: float | None = None,
):
    '''
    Computes the F-test significance of two fits with different numbers of independent parameters.
    Only applies when the Fourier transform and fit ranges (and thus, the number of independent data points) are the same.
    
    :param null_r: Crystallographic R-factor (R^2) of the better (lower-R-factor) fit.
    :type null_r: float
    :param alternative_r: Crystallographic R-factor of the alternative fit.
    :type alternative_r: float
    :param null_parameters: Number of independent parameters in the null model.
    :type null_parameters: float
    :param alternative_parameters: Number of independent parameters in the alternative model.
    :type alternative_parameters: float
    :param n_independent: Number of independent data points.
    :type n_independent: float
    :param b: Description
    :type b: float | None
    '''
    r = null_r / alternative_r
    a = (n_independent - null_parameters) * 0.5
    # b is the total number of free parameters, not the net number of free parameters
    if b is None:
        b = (null_parameters - alternative_parameters) * 0.5
        
    return 1 - betainc(a, b, r)


def feffit_result_hamilton_f(
    null_feffit_result,
    alternative_feffit_result,
    b: float | None = None,
):
    '''
    Wrapper for `hamilton_f`.
    
    :param null_feffit_result: Larch result object of the better (lower-R-factor) fit.
    :param alternative_feffit_result: Larch result object of the alternative fit.
    '''
    null_r = null_feffit_result.rfactor
    null_parameters = null_feffit_result.nvarys
    n_independent = null_feffit_result.n_independent
    alternative_r = alternative_feffit_result.rfactor
    alternative_parameters = alternative_feffit_result.nvarys
    return hamilton_f(null_r, alternative_r, null_parameters, alternative_parameters, n_independent, b=b)

def einstein_model(temperature_K, einstein_temperature_K, static_disorder, mass_1_amu, mass_2_amu):
    reduced_mass_amu = mass_1_amu * mass_2_amu / (mass_1_amu + mass_2_amu)
    return EINS_FACTOR/(reduced_mass_amu * einstein_temperature_K * np.tanh(0.5 * einstein_temperature_K / temperature_K)) + static_disorder

def fit_einstein_model(temperature_K, sigma_A2, mass_1_amu, mass_2_amu, sigma_A2_error=None, parameter_hints: dict | None = None):
    DEFAULT_PARAMETER_HINTS = {
            'einstein_temperature_K': {
                'value': 300.0,
                'min': 0.0,
                'max': 999.9,
            },
            'static_disorder': {
                'value': 0.0005,
                'min': 0.0,
                'max': 0.05,
            },
        }
    
    if parameter_hints is None:
        parameter_hints = DEFAULT_PARAMETER_HINTS
    
    model = Model(einstein_model, independent_vars=['temperature_K', 'mass_1_amu', 'mass_2_amu'])
    for parameter in DEFAULT_PARAMETER_HINTS:
        if parameter not in parameter_hints:
            parameter_hints[parameter] = DEFAULT_PARAMETER_HINTS[parameter]
            
        model.set_param_hint(parameter, **parameter_hints[parameter])
        
    parameters = model.make_params()
    
    weights = None
    if sigma_A2_error is not None:
        weights = 1.0 / np.asarray(sigma_A2_error)
    
    return model.fit(sigma_A2, params=parameters, temperature_K=temperature_K, mass_1_amu=mass_1_amu, mass_2_amu=mass_2_amu, weights=weights)
            
            
def sort_by_column(arr, sort_index: int = 0):
    return arr[arr[:, sort_index].argsort(), :]

def group_to_ascii(
    group,
    name,
    dir,
    export_rs=True,
    export_ks=True,
    export_es=True,
    **kwargs
):
    if 'index' not in kwargs:
        kwargs['index'] = False
    
    if export_rs:
        A = np.array([
            group.r,
            group.chir_mag,
            group.chir_re,
            group.chir_im,
        ]).transpose()
        A = pd.DataFrame(
            A,
            columns=['r', 'data_chir_mag', 'data_chir_re', 'data_chir_im']
        )
        A.to_csv(dir / f'{name}_rs.dat', **kwargs)
    
    if export_ks:
        B = np.array([
            group.k,
            group.chi,
        ]).transpose()
        B = pd.DataFrame(
            B,
            columns=['k', 'data_chi']
        )
        B.to_csv(dir / f'{name}_ks.dat', **kwargs)
    
    if export_es:
        C = np.array([
            group.energy,
            group.mu,
            group.norm,
        ]).transpose()
        C = pd.DataFrame(
            C,
            columns=['energy', 'mu', 'norm']
        )
        C.to_csv(dir / f'{name}_es.dat', **kwargs)

def dset_to_ascii(
    dset,
    name,
    dir,
    **kwargs
):
    if 'index' not in kwargs:
        kwargs['index'] = False
    
    A = np.array([
        dset.data.r,
        dset.data.chir_mag,
        dset.data.chir_re,
        dset.data.chir_im,
        dset.model.chir_mag,
        dset.model.chir_re,
        dset.model.chir_im,
    ]).transpose()
    A = pd.DataFrame(
        A,
        columns=['r', 'data_chir_mag', 'data_chir_re', 'data_chir_im', 'model_chir_mag', 'model_chir_re', 'model_chir_im']
    )
    
    B = np.array([
        dset.data.k,
        dset.data.chi,
        dset.model.chi,
    ]).transpose()
    B = pd.DataFrame(
        B,
        columns=['k', 'data_chi', 'model_chi']
    )

    A.to_csv(dir / f'{name}_rs.dat', **kwargs)
    B.to_csv(dir / f'{name}_ks.dat', **kwargs)
            

def parse_feff(
    std_dir,
    paths_dat='paths.dat',
    renfeff_suffix='.f8',
):
    """Parse FEFF ``paths.dat`` and include related single-scattering paths.

    Returns
    -------
    pandas.DataFrame
        Columns: ``index``, ``nleg``, ``degeneracy``, ``r``, ``ss paths``,
        ``filename``, ``renfeff filename``.
    """
    path = pathlib.Path(std_dir) / paths_dat
    renfeff_suffix = str(renfeff_suffix)
    if len(renfeff_suffix) > 0 and (not renfeff_suffix.startswith('.')):
        renfeff_suffix = f'.{renfeff_suffix}'

    with open(path, 'r') as f:
        lines = f.readlines()

    entries = []
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i]
        if 'index, nleg, degeneracy, r=' not in line:
            i += 1
            continue

        lhs, rhs = line.split('r=', maxsplit=1)
        tokens = lhs.split()
        if len(tokens) < 3:
            i += 1
            continue

        index = int(tokens[0])
        nleg = int(tokens[1])
        degeneracy = float(tokens[2])
        r = float(rhs.strip())

        atoms = []
        j = i + 1
        while j < n_lines and len(atoms) < nleg:
            # stop at the next entry header if this entry is malformed
            if 'index, nleg, degeneracy, r=' in lines[j]:
                break

            atom_tokens = lines[j].split()
            # atom rows begin with x, y, z and contain a quoted label
            if len(atom_tokens) >= 5 and ("'" in lines[j]):
                try:
                    float(atom_tokens[0])
                    float(atom_tokens[1])
                    float(atom_tokens[2])
                    ipot = int(atom_tokens[3])
                except ValueError:
                    j += 1
                    continue

                label_start = lines[j].find("'")
                label_end = lines[j].find("'", label_start + 1)
                if label_start != -1 and label_end != -1:
                    label = lines[j][label_start + 1:label_end].strip()
                    tail_tokens = lines[j][label_end + 1:].split()
                    if len(tail_tokens) == 0:
                        j += 1
                        continue
                    try:
                        rleg = float(tail_tokens[0])
                    except ValueError:
                        j += 1
                        continue

                    atoms.append((ipot, label, round(rleg, 6)))
            j += 1

        entries.append(
            {
                'index': index,
                'nleg': nleg,
                'degeneracy': degeneracy,
                'r': r,
                'atoms': atoms,
            }
        )
        i = j

    ss_entries = []
    for entry in entries:
        if entry['nleg'] != 2:
            continue

        scatterers = [(label.upper(), rleg) for ipot, label, rleg in entry['atoms'] if ipot != 0]
        if len(scatterers) == 0:
            continue

        # nleg==2 has one scatterer; use path header r as the matching distance
        ss_entries.append((entry['index'], scatterers[0][0], round(entry['r'], 6)))

    for entry in entries:
        entry_atoms = Counter((label.upper(), rleg) for ipot, label, rleg in entry['atoms'] if ipot != 0)
        related = []
        for ss_index, ss_label, ss_r in ss_entries:
            if entry_atoms[(ss_label, ss_r)] > 0:
                related.append(ss_index)
        related.sort()
        entry['ss paths'] = related

        entry['filename'] = f"feff{entry['index']:04d}.dat"

        absorber_labels = [label for ipot, label, _ in entry['atoms'] if ipot == 0]
        scatterer_labels = sorted([label for ipot, label, _ in entry['atoms'] if ipot != 0])
        if len(absorber_labels) > 0:
            atom_labels = [absorber_labels[0]] + scatterer_labels
        else:
            atom_labels = sorted([label for _, label, _ in entry['atoms']])

        renfeff_stem = f"{'_'.join(atom_labels)}_{entry['r']:.4f}"
        entry['renfeff filename'] = f"{renfeff_stem}{renfeff_suffix}"

    return pd.DataFrame(
        [
            {
                'index': entry['index'],
                'nleg': entry['nleg'],
                'degeneracy': entry['degeneracy'],
                'r': entry['r'],
                'ss paths': entry['ss paths'],
                'filename': entry['filename'],
                'renfeff filename': entry['renfeff filename'],
            }
            for entry in entries
        ],
        columns=['index', 'nleg', 'degeneracy', 'r', 'ss paths', 'filename', 'renfeff filename']
    )


class Parsefeff:
    
    @staticmethod
    def parse_feff(
        std_dir,
        renfeff_log='renfeff.log',
        files_dat='files.dat',
        skiprows=18
    ):
        renfeff_df = pd.read_csv(std_dir / renfeff_log, header=None)
        renfeff_df.mask(~renfeff_df[0].str.contains('copying'), inplace=True)
        renfeff_df.dropna(inplace=True)
        renfeff_df = renfeff_df[0].str.split(' ', expand=True)
        renfeff_df.drop([0, 2], axis=1, inplace=True)
        renfeff_df.set_index(1, inplace=True)
        renfeff_df.index = [int(x[4:8]) for x in renfeff_df.index.values]
        
        files_df = pd.read_csv(std_dir / files_dat, sep=r'\s+', skiprows=skiprows, header=None)
        files_df.set_index(0, inplace=True)
        files_df.index = [int(x[4:8]) for x in files_df.index.values]
        
        info_df = pd.concat([files_df, renfeff_df], axis=1)
        info_df.columns = [
            'sig2',
            'amp_ratio',
            'deg',
            'nlegs',
            'reff',
            'path'
        ]
        return info_df

    @staticmethod
    def copy_paths(
        df,
        ks_dir,
        std_dir,
        col='path',
        quiet=True
    ):
        """
        Copy files from std_dir to ks_dir.
        """
        for f in df[col]:
            p = std_dir / f
            target = ks_dir / f
            if target.exists():
                if not quiet:
                    warnings.warn('File {} already exists. Overwriting.'.format(target))
                target.unlink()
            target.hardlink_to(p)

    @staticmethod
    def edit_list(
        sink_dir,
        sink_df,
        list_dat='list.dat',
        list_bak='list.dat.bak',
    ):
        target = sink_dir / list_dat
        backup = sink_dir / list_bak
        try:
            backup.hardlink_to(target)
        except:
            pass
        sink_list = pd.read_csv(target, sep=r'\s+', header=None, skiprows=3, index_col=0)
        sink_list_trimmed = sink_list.loc[sink_list.index.isin(sink_df.index)]
        target.unlink()
        with open(target, 'w') as f:
            f.write('PATH  Rmax= 7.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%\n')
            f.write(' -----------------------------------------------------------------------\n')
            f.write('  pathindex     sig2   amp ratio    deg    nlegs  r effective\n')
            for i in range(len(sink_list_trimmed)):
                f.write('   {}   {}   {}   {}   {}\n'.format(
                    sink_list_trimmed.index[i],
                    sink_list_trimmed.iloc[i, 0],
                    sink_list_trimmed.iloc[i, 2],
                    int(sink_list_trimmed.iloc[i, 3]),
                    sink_list_trimmed.iloc[i, 4],
                ))
        return sink_list_trimmed

    @staticmethod
    def gen_sink_df(
        info_df,
        explicit_df
    ):
        return info_df.loc[~info_df.index.isin(explicit_df.index)]

    @staticmethod
    def get_scatterers(
        path,
        skiprows=18
    ):
        df = pd.read_csv(path, skiprows=skiprows)
        # remove leading whitespace, then find the index of the first row after the table of scatterers
        idx = df.index[df.iloc[:, 0].str.lstrip().str.startswith('k')]
        df = df.iloc[:idx[0], :]
        # split the header into column names
        columns = df.columns[0].split()
        columns += ['at']
        df = df.iloc[:, 0].str.split(expand=True)
        df = df.iloc[:, :6]
        df.columns = columns
        # drop the absorbing atom
        df.drop(axis=0, index=0, inplace=True)
        return df

    @staticmethod
    def check_row_match(df1, df2):
        # Perform an inner merge to find matching rows
        matching_rows = pd.merge(df1, df2, how='inner')
        return not matching_rows.empty

    @classmethod
    def find_paths(
        cls,
        path,
        std_dir,
        ext='f8',
        **kwargs
    ):
        paths = sorted(std_dir.glob('*.{}'.format(ext)))
        path_df_dict = {}
        for f in paths:
            path_df_dict[f.stem + '.{}'.format(ext)] = cls.get_scatterers(f, **kwargs)
        keys_list = []
        for k, df in path_df_dict.items():
            if cls.check_row_match(path_df_dict[path], df):
                keys_list.append(k)
        return keys_list