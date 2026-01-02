import copy
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.signal import decimate, resample, savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.special import betainc

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
        import_data_kwargs=None,
        **kwargs
    ):
        if import_data_kwargs is None:
            import_data_kwargs = {}
        if 'E_shift' not in import_data_kwargs.keys():
            import_data_kwargs['E_shift'] = E_shift
        if data_list is not None:
            self.data_list = data_list
        else:
            self.path_generator(dir, *args, **kwargs)
        self.make_arr_list(**import_data_kwargs)
        # self.arr_list = []
        # for f in self.data_list:
        #     self.arr_list.append(self.import_data(f, **import_data_kwargs))
    
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
        
        # find start of data
        skiprows = df[df.iloc[:, 0].str.contains('Offset')].index.values[0] + 1
        
        # slice and dice
        raw_arr = df.iloc[skiprows:, 0].str.split().apply(pd.to_numeric).apply(pd.Series).to_numpy()
        arr = np.empty((len(raw_arr), 2))
        # return mu vs E
        arr[:, 0] = self.energy(raw_arr[:, 1], d_spacing) + E_shift
        arr[:, 1] = -np.log(raw_arr[:, -1]/raw_arr[:, -2])
        arr = arr[arr[:, 0].argsort(), :]
        return arr
    
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
        **kwargs
    ):
        if 'label' not in kwargs.keys():
            kwargs['label'] = 'energy mu norm'
        lio.write_ascii(
            output_file, 
            self.group.energy, 
            self.group.mu, 
            self.group.norm, 
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
        kwargs['kind'] = kind
        if grid_points is None:
            # attempt to use min and max E from first array
            if bounds is None:
                bounds = [
                    np.ceil(min(self.arr_list[0][:, 0]/step))*step,
                    np.floor(max(self.arr_list[0][:, 0]/step))*step
                ]
            grid_points = np.arange(bounds[0], bounds[1], step)
        self.mu_interp = np.empty((len(grid_points), len(self.arr_list)))
        self.arr_avg = np.empty((len(grid_points), 2))
        for i, arr in enumerate(self.arr_list):
            self.mu_interp[:, i] = interp1d(
                remove_dups(arr[:, 0], tiny=tiny),
                arr[:, 1],
                grid_points,
                **kwargs
            )
            # self.mu_interp[:, i] = self.interpolate_and_resample(
            # arr, bounds=bounds, step=step, grid_points=grid_points, s=s
            # )[:, 1]
        self.arr_avg[:, 0] = grid_points
        self.arr_avg[:, 1] = np.average(self.mu_interp, axis=1)
        self.arr_avg = self.arr_avg[~np.isnan(self.arr_avg).any(axis=1)]
        
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
        ax.plot(self.arr_avg[:, 0], self.arr_avg[:, 1])
        try:
            ax.plot(self.group.energy, self.group.mu, 'k--')
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
        self.feffit_outputs = []
    
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
    
    def plot_kr_fitted(
        self,
        keys=None,
        feffit_run_outputs: dict | None = None,
        feffit_outputs_index=-1,
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
        DEFAULT_DATA_COLOR = 'k'
        DEFAULT_DATA_LINESTYLE = '-'
        DEFAULT_MODEL_COLOR = '#4298B5'
        DEFAULT_MODEL_LINESTYLE = '--'
        DEFAULT_FONTSIZE = 12
        
        if k_plot_data_kwargs is None:
            k_plot_data_kwargs = {}
            
        if r_plot_data_kwargs is None:
            r_plot_data_kwargs = {}
            
        if k_plot_model_kwargs is None:
            k_plot_model_kwargs = {}
            
        if r_plot_model_kwargs is None:
            r_plot_model_kwargs = {}
            
        if k_ax_opts_kwargs is None:
            k_ax_opts_kwargs = {}
            
        if r_ax_opts_kwargs is None:
            r_ax_opts_kwargs = {}
            
        if 'color' not in k_plot_data_kwargs:
            k_plot_data_kwargs['color'] = DEFAULT_DATA_COLOR
            
        if 'color' not in k_plot_model_kwargs:
            k_plot_model_kwargs['color'] = DEFAULT_MODEL_COLOR
            
        if 'color' not in r_plot_data_kwargs:
            r_plot_data_kwargs['color'] = DEFAULT_DATA_COLOR
            
        if 'color' not in r_plot_model_kwargs:
            r_plot_model_kwargs['color'] = DEFAULT_MODEL_COLOR
            
        if 'linestyle' not in k_plot_data_kwargs:
            k_plot_data_kwargs['linestyle'] = DEFAULT_DATA_LINESTYLE
            
        if 'linestyle' not in k_plot_model_kwargs:
            k_plot_model_kwargs['linestyle'] = DEFAULT_MODEL_LINESTYLE
            
        if 'linestyle' not in r_plot_data_kwargs:
            r_plot_data_kwargs['linestyle'] = DEFAULT_DATA_LINESTYLE
            
        if 'linestyle' not in r_plot_model_kwargs:
            r_plot_model_kwargs['linestyle'] = DEFAULT_MODEL_LINESTYLE
            
        if 'fontsize' not in k_ax_opts_kwargs:
            k_ax_opts_kwargs['fontsize'] = DEFAULT_FONTSIZE
            
        if 'fontsize' not in r_ax_opts_kwargs:
            r_ax_opts_kwargs['fontsize'] = DEFAULT_FONTSIZE
            
        
        if feffit_run_outputs is None:
            feffit_run_outputs = self.feffit_outputs[feffit_outputs_index]
            
        if keys is None:
            keys = feffit_run_outputs.keys()
            
        fig_ax_outputs = {}
        for k in keys:
            v = feffit_run_outputs[k]
            dset, _ = v
            rmin = dset.transform.rmin
            rmax = dset.transform.rmax
            fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained', sharex='col', sharey='col')
            axs[0].plot(dset.data.k, dset.data.chi*dset.data.k**k_weight, **k_plot_data_kwargs)
            axs[0].plot(dset.model.k, dset.model.chi*dset.data.k**k_weight, **k_plot_model_kwargs)
            axs[1].plot(dset.data.r, dset.data.chir_mag, label="Data", **r_plot_data_kwargs)
            axs[1].plot(dset.data.r, dset.data.chir_re, **r_plot_data_kwargs)
            axs[1].plot(dset.model.r, dset.model.chir_mag, **k_plot_model_kwargs)
            axs[1].plot(dset.model.r, dset.model.chir_re, **k_plot_model_kwargs)
            if plot_text:
                axs[1].text(
                    0.95,
                    0.95,
                    k,
                    transform=axs[1].transAxes,
                    ha='right',
                    va='top',
                    fontsize=r_ax_opts_kwargs['fontsize'],
                )
                
            if plot_fit_window:
                axs[1].add_patch(
                    patches.Rectangle(
                        (rmin, -30),
                        rmax - rmin,
                        99,
                        color=k_plot_model_kwargs['color'],
                        alpha=0.1,
                        zorder=0
                    )
                )
                
            Plot.ax_opts(
                axs[0],
                **k_ax_opts_kwargs,
            )
            Plot.ax_opts(
                axs[1],
                **r_ax_opts_kwargs,
            )
            
            if legend:
                custom_lines = [
                    Line2D([0], [0], color=k_plot_data_kwargs['color'], linestyle=k_plot_data_kwargs['linestyle'], label=f'Data'),
                    Line2D([0], [0], color=k_plot_model_kwargs['color'], linestyle=k_plot_model_kwargs['linestyle'], label=f'Fit')
                ]
                legend_kwargs = {
                    'frameon': False,
                    'fontsize': k_ax_opts_kwargs['fontsize'],
                    'labelspacing': 0.25,
                    'handlelength': 1.2
                }
                axs[0].legend(handles=custom_lines, 
                    loc='lower right', 
                    **legend_kwargs
                )
            
                custom_lines = [
                    Line2D([0], [0], color=r_plot_data_kwargs['color'], linestyle=r_plot_data_kwargs['linestyle'], label=f'Data'),
                    Line2D([0], [0], color=r_plot_model_kwargs['color'], linestyle=r_plot_model_kwargs['linestyle'], label=f'Fit')
                ]
                legend_kwargs['fontsize'] = r_ax_opts_kwargs['fontsize']
                axs[1].legend(handles=custom_lines, 
                    loc='lower right', 
                    **legend_kwargs
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
        group=None
    ):
        if group is None:
            group = self.groups[key]
            
        if autobk_kwargs is not None:
            lx.autobk(group.energy, group.norm, group=group, **autobk_kwargs)
            
        k_transformed_data = lx.feffit_transform(**xftf_kwargs)
        feffit_dataset = lx.feffit_dataset(data=group, pathlist=feff_paths, transform=k_transformed_data)
        feffit_output = lx.feffit(parameter_group, [feffit_dataset], method=method)
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
    ):
        """
        Fits the FEFF paths specified by feff_paths, parameterized by parameter_groups.
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
        if feffit_run_index is None or (feffit_run_index > len(self.feffit_outputs)):
            self.feffit_outputs.append({})
            feffit_run_index = len(self.feffit_outputs)
        if len(self.feffit_outputs) == 0:
            self.feffit_outputs = [{}]
            
        for k in keys:
            feffit_run_outputs[k] = self.feffit_single(
                k,
                feff_paths,
                parameter_group,
                autobk_kwargs=autobk_kwargs[k],
                xftf_kwargs=xftf_kwargs[k],
                method=method,
            )
            if file_name_prefix is not None:
                file_name = str(file_name_prefix) + '_'
            else:
                file_name = ''
                
            if save_directory is not None:
                file_name += f"{k}_run{feffit_run_index}.txt"
                with open(pathlib.Path(save_directory) / file_name, 'w') as f:
                    f.write(lx.feffit_report(feffit_run_outputs[k][1]))
            
        self.feffit_outputs[feffit_run_index-1] = feffit_run_outputs
            
        return feffit_run_outputs
    
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
            
        return feffit_run_outputs, group, xftf_kwargs
    
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
    axs[2].vlines(group.rbkg, -999, 999)
    axs[2].set_xlim(0, 6)
    fig.set_size_inches(9, 3)
    return fig, axs

def autobk_xftf(
    k,
    group,
    autobk_kwargs=None,
    xftf_kwargs=None
):
    if autobk_kwargs is None:
        autobk_kwargs = {}
    if xftf_kwargs is None:
        xftf_kwargs = {}
    lx.autobk(group.energy, group.norm, group=group, **autobk_kwargs)
    lx.xftf(group.k, group.chi, group=group, **xftf_kwargs)
    return k, group


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