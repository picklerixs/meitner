import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.signal import decimate, resample, savgol_filter
from scipy.interpolate import UnivariateSpline

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
        if df:
            self.df = df
        elif file:
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
        **kwargs
        ):
        if not (fig and ax):
            fig, ax = plt.subplots(layout='constrained')
        
        if plot_fit:
            ax.plot(df[0], df[5], color=color, zorder=999, linewidth=linewidth, linestyle='--')
            ax.plot(df[0], df[6], color=color, zorder=998, linewidth=linewidth, linestyle='--')
            if errorbar:
                ax.errorbar(df[0], df[1], yerr=df[2], fmt='-', color='black')
                ax.errorbar(df[0], df[3], yerr=df[4], fmt='-', color='black')
            else:
                ax.plot(df[0], df[1], color='black')
                ax.plot(df[0], df[3], color='black')
        else:
            if errorbar:
                ax.errorbar(df[0], df[1], yerr=df[2], fmt='-', color=color)
                ax.errorbar(df[0], df[3], yerr=df[4], fmt='-', color=color)
            else:
                ax.plot(df[0], df[1], color=color, linestyle='-')
                ax.plot(df[0], df[3], color=color, linestyle='-')
                
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