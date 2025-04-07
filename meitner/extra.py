import numpy as np
import pandas as pd

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.interpolate import UnivariateSpline # default is cubic spline


class Aux:
    '''
    Aux methods.
    '''
    @classmethod
    def return_entry(cls, x, k):
        if cls.is_list_or_tuple(x):
            return x[k]
        else:
            return x
        
    @staticmethod
    def is_list_or_tuple(x):
        return isinstance(x, list) or isinstance(x, tuple)
    
    @staticmethod
    def is_float_or_int(x):
        return isinstance(x, float) or isinstance(x, int)
    
    @classmethod
    def encapsulate(cls, x):
        if cls.is_float_or_int(x[0]):
            return [x]
        else:
            return x
        
    @classmethod
    def initialize_dict(cls, x, dict_keys, len_list=1, ref_dict=None):         
        if (cls.is_float_or_int(x) or (x is None)) and isinstance(ref_dict, dict):
            x_dict = {}
            for dk in dict_keys:
                x_dict.update({dk: [x for _ in range(len(ref_dict[dk]))]})
            return x_dict
        
        if (cls.is_float_or_int(x) or (isinstance(x, bool)) or (x is None)):
            x = [x for _ in range(len_list)]
        
        if cls.is_list_or_tuple(x):
            x_dict = {}
            x_dict.update({dk: x for dk in dict_keys})
        elif isinstance(x, dict):
            x_dict = x
        return x_dict
     
    @staticmethod
    def ax_opts(ax, xlim=None, ylim=None, xticks=False, yticks=False, tick_direction='out',
                major_tick_multiple=0, minor_tick_multiple=0):
        if xlim != None:
            ax.set_xlim([min(xlim),max(xlim)])
        if ylim != None:
            ax.set_ylim(ylim)
            
        if xticks != False:
            ax.set_xticks(xticks)
        if yticks == False:
            ax.set_yticks([])
        elif yticks == True:
            ax.set_yticks()
        else:
            ax.set_yticks(yticks)
            
        ax.tick_params(direction=tick_direction)
        # specifying major_tick_multiple overrides manual xticks spec
        if major_tick_multiple > 0:
            ax.xaxis.set_major_locator(MultipleLocator(major_tick_multiple))
        if minor_tick_multiple > 0:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_multiple))


class Interp:
    @staticmethod
    def interpolate_df(df, column, bounds, step):
        """
        Use splines to reevaluate a DataFrame on a specified grid for a given column.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to reevaluate.
        column (str): The column name to base the reevaluation on.
        grid_points (array-like): The grid points to reevaluate the DataFrame on.
        
        Returns:
        pd.DataFrame: The reevaluated DataFrame.
        """
        grid_points = np.arange(bounds[0], bounds[1], step)
        # Sort the DataFrame based on the specified column
        df_sorted = df.sort_values(by=column)
        
        # Create a dictionary to store the reevaluated data
        reevaluated_data = {column: grid_points}
        
        # Use splines to reevaluate each column in the DataFrame
        for col in df_sorted.columns:
            if col != column:
                spline = UnivariateSpline(df_sorted[column], df_sorted[col], s=0)
                reevaluated_data[col] = spline(grid_points)
        
        # Create a new DataFrame with the reevaluated data
        df_reevaluated = pd.DataFrame(reevaluated_data)
        
        return df_reevaluated

    @classmethod
    def interpolate_dfs(cls, dfs, bounds, column='B.E.', step=0.1):
        data_interpolated = []
        for df in dfs:
            df_interpolated = cls.interpolate_df(df, column, bounds, step)
            data_interpolated.append(df_interpolated)
        return data_interpolated

    @staticmethod
    def avg_dfs(dfs):
        df_avg = dfs[0].copy()
        for df in dfs[1:]:
            df_avg += df
        df_avg /= len(dfs)
        return df_avg

    @classmethod
    def interpolate_avg_dfs(cls, dfs, bounds, column='B.E.', step=0.1):
        data_interpolated = cls.interpolate_dfs(dfs, bounds, column=column, step=step)
        df_avg = cls.avg_dfs(data_interpolated)
        return data_interpolated, df_avg
    
    
class Plot:
    '''
    Methods for plot styling
    '''
    def __init__(
        self
    ):
        pass
    
    
    @staticmethod
    def set_rc(
        font_families=['Arial','Noto Sans'],
        usetex=False
    ):
        rc('font',**{'family':'sans-serif','sans-serif':font_families})
        rc('text', usetex=usetex)
    
    
    @staticmethod
    def set_labels(
        ax,
        xlabel, 
        ylabel, 
        fontsize, 
        **kwargs
    ):
        ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
        ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)
    
    
    @staticmethod
    def sample_colormap(
        cmap,
        ncolors,
        start=0,
        reverse=False
    ):
        cmap = getattr(plt.cm, cmap)
        ncolors = min(cmap.N, ncolors)
        colors = [cmap(int(x*cmap.N/ncolors)) for x in range(start,start+ncolors,1)]
        return colors[::-1] if reverse else colors
    
    
    def ax_opts(
        self,
        ax, 
        # quantitative axis settings
        xlim=None, 
        ylim=None, 
        xticks=None, 
        yticks=None, 
        xmajtm=None, 
        xmintm=None,
        ymajtm=None,
        ymintm=None,
        # line styling
        axes_linewidth=1.35,
        tick_linewidth=None,
        tick_length=None,
        tick_direction='out',
        # label settings
        xlabel=None,
        ylabel=None,
        fontsize=12,
        label_preset=None,
    ):
        # quantitative axis settings
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        # specifying major_tick_multiple overrides manual xticks spec
        if xmajtm:
            ax.xaxis.set_major_locator(MultipleLocator(xmajtm))
        if xmintm:
            ax.xaxis.set_minor_locator(MultipleLocator(xmintm))    
        if ymajtm:
            ax.yaxis.set_major_locator(MultipleLocator(ymajtm))
        if ymintm:
            ax.yaxis.set_minor_locator(MultipleLocator(ymintm))
            
        # line styling
        if axes_linewidth:
            rcParams['axes.linewidth'] = axes_linewidth
        if not tick_linewidth:
            tick_linewidth = axes_linewidth*0.9
        if not tick_length:
            tick_length = axes_linewidth*5
        ax.tick_params(
            direction=tick_direction,
            width=tick_linewidth,
            length=tick_length,
            labelsize=fontsize,
            axis='both',
            which='both'
        )
        ax.tick_params(
            length=tick_length*0.5,
            axis='both',
            which='minor'
        )
            
            
class Plot:
    '''
    Methods for plot styling
    '''
    def __init__(
        self
    ):
        pass
    
    
    @staticmethod
    def set_rc(
        font_families=['Arial','Noto Sans'],
        usetex=False
    ):
        rc('font',**{'family':'sans-serif','sans-serif':font_families})
        rc('text', usetex=usetex)
    
    
    @staticmethod
    def set_labels(
        ax,
        xlabel, 
        ylabel, 
        fontsize, 
        **kwargs
    ):
        ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
        ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)
    
    
    @staticmethod
    def sample_colormap(
        cmap,
        ncolors,
        start=0,
        reverse=False
    ):
        cmap = getattr(plt.cm, cmap)
        ncolors = min(cmap.N, ncolors)
        colors = [cmap(int(x*cmap.N/ncolors)) for x in range(start,start+ncolors,1)]
        return colors[::-1] if reverse else colors
    
    @classmethod
    def ax_opts(
        cls,
        ax, 
        # quantitative axis settings
        xlim=None, 
        ylim=None, 
        xticks=None, 
        yticks=None, 
        xmajtm=None, 
        xmintm=None,
        ymajtm=None,
        ymintm=None,
        # line styling
        axes_linewidth=1.35,
        tick_linewidth=None,
        tick_length=None,
        tick_direction='out',
        # label settings
        xlabel=None,
        ylabel=None,
        fontsize=12,
        label_preset=None,
    ):
        # quantitative axis settings
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        # specifying major_tick_multiple overrides manual xticks spec
        if xmajtm:
            ax.xaxis.set_major_locator(MultipleLocator(xmajtm))
        if xmintm:
            ax.xaxis.set_minor_locator(MultipleLocator(xmintm))    
        if ymajtm:
            ax.yaxis.set_major_locator(MultipleLocator(ymajtm))
        if ymintm:
            ax.yaxis.set_minor_locator(MultipleLocator(ymintm))
            
        # line styling
        if axes_linewidth:
            rcParams['axes.linewidth'] = axes_linewidth
        if not tick_linewidth:
            tick_linewidth = axes_linewidth*0.9
        if not tick_length:
            tick_length = axes_linewidth*5
        ax.tick_params(
            direction=tick_direction,
            width=tick_linewidth,
            length=tick_length,
            labelsize=fontsize,
            axis='both',
            which='both'
            )
        ax.tick_params(
            length=tick_length*0.5,
            axis='both',
            which='minor'
        )
        
        # label settings
        if label_preset == 'ks_exafs':
            cls.set_labels(
                ax,
                xlabel='$k$ (Å$^{-1}$)',
                ylabel=r"$k^{3}\chi(k)$",
                fontsize=fontsize
            )
        elif label_preset == 'rs_exafs':
            cls.set_labels(
                ax,
                xlabel='$r$ (Å)',
                ylabel=r"Fourier Transform of $k^{3}\chi(k)$",
                fontsize=fontsize
            )
        elif label_preset == 'xanes':
            cls.set_labels(
                ax,
                xlabel='Photon Energy (eV)',
                ylabel=r"Normalized Absorbance",
                fontsize=fontsize
            )
        elif label_preset == 'xps':
            cls.set_labels(
                ax,
                xlabel='Binding Energy (eV)',
                ylabel=r"Normalized Intensity",
                fontsize=fontsize
            )
        elif label_preset == 'xaes':
            cls.set_labels(
                ax,
                xlabel='Kinetic Energy (eV)',
                ylabel=r"Normalized Intensity",
                fontsize=fontsize
            )
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        # label settings
        if label_preset == 'ks_exafs':
            cls.set_labels(
                ax,
                xlabel='$k$ (Å$^{-1}$)',
                ylabel=r"$k^{3}\chi(k)$",
                fontsize=fontsize
            )
        elif label_preset == 'rs_exafs':
            cls.set_labels(
                ax,
                xlabel='$r$ (Å)',
                ylabel=r"Fourier Transform of $k^{3}\chi(k)$",
                fontsize=fontsize
            )
        elif label_preset == 'xanes':
            cls.set_labels(
                ax,
                xlabel='Photon Energy (eV)',
                ylabel=r"Normalized Absorbance",
                fontsize=fontsize
            )
        elif label_preset == 'xps':
            cls.set_labels(
                ax,
                xlabel='Binding Energy (eV)',
                ylabel=r"Normalized Intensity",
                fontsize=fontsize
            )
        elif label_preset == 'xaes':
            cls.set_labels(
                ax,
                xlabel='Kinetic Energy (eV)',
                ylabel=r"Normalized Intensity",
                fontsize=fontsize
            )
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)