import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from math import ceil
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from matplotlib import rc, rcParams
from matplotlib.lines import Line2D

from .extra import Aux, Plot


class Casa:
    
    fontsize=12
    labelsize=fontsize
    legendsize=fontsize
    linewidth=1.25

    font_family='Arial'
    axes_linewidth=1.35
    tick_linewidth=axes_linewidth*.9
    tick_length=tick_linewidth*5
    marker_size=7
    marker_edge_width=linewidth/2
    labelpad=5

    rc('font',**{'family':'sans-serif','sans-serif':[font_family]})
    rc('text', usetex=False)
    rcParams["mathtext.default"] = 'regular'
    rcParams['axes.linewidth'] = axes_linewidth

    @staticmethod
    def load_csv(path, norm='area', skiprows=7, norm_target='Envelope', norm_point=None, norm_comp=None):
        di = pd.read_csv(path, skiprows=skiprows)
        di = di.dropna(axis=1)
        comps = []
        norm_suffix = ''
        n_cols = len(di.columns.values)
        if n_cols > 4:
            n_comps = int((len(di.columns.values) - 8)/2)
        else:
            n_comps = 0
        if n_comps > 0:
            for i in range(n_comps):
                comps += ['p{} CPS'.format(i)]
                di.columns.values[i+2] = 'p{}'.format(i)
                di.columns.values[i+6+n_comps] = 'p{} CPS'.format(i)
        comps += ['CPS']
        
        bg_flag = 'Background CPS' in di.columns.values
        comps_flag = n_comps > 0
        if bg_flag:
            comps += ['Background CPS']
            norm_suffix = '_no_bg'
        if comps_flag:
            comps += ['Envelope CPS']
            
        if bg_flag:
            for c in comps:
                di[c+'_no_bg'] = di[c] - di['Background CPS']
            min_cps = min(di['Background CPS'+norm_suffix])
        else:
            min_cps = min(di['CPS'+norm_suffix])
        if norm == 'area':
            denom = -trapezoid(di['CPS'+norm_suffix], x=di['B.E.'])
        elif norm == 'minmax':
            denom = max(di['CPS'+norm_suffix])-min_cps
        elif norm == 'point' and (isinstance(norm_point, float) or isinstance(norm_point, int)):
            result_index = di['B.E.'].sub(norm_point).abs().idxmin()
            denom = di['CPS'+norm_suffix][result_index]
        elif norm == 'comp' and isinstance(norm_comp, int):
            denom = -trapezoid(di['p{} CPS'.format(norm_comp)], x=di['B.E.'])
        elif norm[0] == 'p':
            denom = -trapezoid(di[norm+' CPS'+norm_suffix], x=di['B.E.'])
        for c in comps:
            di[c+'_norm'] = (di[c] - min_cps)/denom
            if norm_suffix == '_no_bg':
                di[c+norm_suffix+'_norm'] = (di[c+norm_suffix] - min_cps)/denom
                
        if n_comps > 0:
            di['residual'] = di['CPS'] - di['Envelope CPS']
            di['residual_norm'] = di['CPS_norm'] - di['Envelope CPS_norm']
        return di

    @classmethod
    def plot_single(cls, di, color, xlim, ylim, savefig=False, dim=[4,3], ylabel='Intensity (a.u.)', subtract_bg=False):
            fig, ax = plt.subplots(layout='constrained')

            if subtract_bg:
                bg_suffix = '_no_bg'
            else:
                bg_suffix = ''
            
            shift = 0
            ax.plot(di['B.E.'], di['CPS{}_norm'.format(bg_suffix)]+shift, linewidth=cls.linewidth, color=color, zorder=999)
            ax.hlines(shift, 0, 999, color='gray', linewidth=cls.linewidth*2/3, zorder=100+1)

            Aux.ax_opts(ax, major_tick_multiple=5, minor_tick_multiple=1, xlim=xlim, ylim=ylim)
            ax.set_ylabel(ylabel, fontsize=cls.fontsize, labelpad=cls.labelpad*2/3)
            ax.set_xlabel('Binding Energy (eV)', fontsize=cls.fontsize, labelpad=cls.labelpad)
            ax.invert_xaxis()

            ax.tick_params(labelsize=cls.fontsize)
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length, which='major')
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length*0.5, which='minor')

            fig.set_size_inches(*dim)
            if savefig:
                    fig.savefig(savefig)
                    
    @classmethod
    def plot_stack(cls, 
        # ARGS
        data, 
        color, 
        xlim, 
        ylim, 
        # KWARGS
        comp_id=None,
        comp_color=None, 
        comp_line=True,
        comp_line_alpha=0.5,
        comp_fill=True,
        comp_fill_alpha=0.25,
        data_color='gray',
        data_style='markers',
        dim=[4,3], 
        envelope_color='match',
        hline=True, 
        legend=True,
        legend_fontsize=None,
        legend_loc='upper right',
        linewidth=None,
        plot_comps=False, 
        plot_envelope=False, 
        plot_residuals=False, 
        residual_color='gray',
        residual_offset=-0.15, 
        savefig=False, 
        scale=1,
        shift=0, 
        subtract_bg=False, 
        top_text=None,
        major_tick_multiple=5, 
        marker='.',
        marker_alpha=0.5,
        minor_tick_multiple=1,
        x_energy='B.E.',
        xlabel=None, 
        ylabel='Intensity (a.u.)',
        yticks=[],
        fontsize=None,
        **kwargs
    ):
        # minimizes clipping and ensures figure conforms to dim
        # more flexible than plt.tightlayout()
        fig, ax = plt.subplots(layout='constrained')
        
        if isinstance(linewidth, int) or isinstance(linewidth, float):
            cls.linewidth = linewidth
            
        if fontsize:
            cls.fontsize=fontsize
        
        if not isinstance(xlabel, str):
            if x_energy == 'K.E.':
                xlabel = 'Kinetic Energy (eV)'
            else:
                xlabel = 'Binding Energy (eV)'

        if subtract_bg:
            bg_suffix = '_no_bg'
        else:
            bg_suffix = ''

        # plot only specified components
        if plot_comps:
            if Aux.is_float_or_int(comp_id) and (comp_id is not None):
                comp_id = [comp_id]
            if Aux.is_list_or_tuple(comp_id):
                comp_id = ['p{}'.format(k) for k in comp_id]

        for i in range(len(data)):
            di = data[i]
            if isinstance(shift, float) or isinstance(shift, int):
                si = shift*i
            else:
                si = shift[i]
            
            # if Aux.is_list_or_tuple(scale):
            #     scale_i = scale[i]
            # else:
            #     scale_i = scale
            try:
                scale_i = scale[i]
            except:
                scale_i = scale
            
            
            if data_color == 'match':
                data_color_i = color[i]
            else:
                data_color_i = data_color
            if residual_color == 'match':
                residual_color_i = color[i]
            else:
                residual_color_i = residual_color
                
            if hline:
                ax.hlines(si, 0, 999, color='gray', linewidth=cls.linewidth*0.75, zorder=-999)
                
            if plot_envelope:
                # plot envelope
                if envelope_color == 'match':
                    envelope_color_i = color[i]
                else:
                    envelope_color_i = envelope_color
                ax.plot(di[x_energy], 
                    di['Envelope CPS{}_norm'.format(bg_suffix)]*scale_i+si, 
                    linewidth=cls.linewidth, 
                    color=envelope_color_i, 
                    zorder=1000
                )
                
            if data_style == 'markers':
                # plot data as points
                ax.plot(
                    di[x_energy], 
                    di['CPS{}_norm'.format(bg_suffix)]*scale_i+si, 
                    alpha=marker_alpha,
                    marker=marker, 
                    linewidth=cls.linewidth, 
                    zorder=200, 
                    color=data_color_i,
                    ms=cls.marker_size, 
                    mew=cls.marker_edge_width
                )
            elif data_style == 'line':
                ax.plot(di[x_energy], di['CPS{}_norm'.format(bg_suffix)]*scale_i+si, linewidth=cls.linewidth, color=color[i], zorder=999)
                
            if plot_comps:
                # get component IDs and number of components
                comp_id_i = [id for id in di.columns.values if id.startswith('p') and id[-1].isdigit()]
                n_comps_i = len(comp_id_i)
                if Aux.is_list_or_tuple(comp_id):
                    comp_id_i = [k for k in comp_id if k in comp_id_i]
                    n_comps_i = int(len(comp_id_i))
                
                if comp_color is None:
                    comp_color_i = None
                elif isinstance(comp_color, list) or isinstance(comp_color, tuple):
                    try: 
                        comp_color[i][0][0]
                        comp_color_i = comp_color[i]
                    except:
                        comp_color_i = None
                elif isinstance(comp_color, dict):
                    comp_color_i = comp_color[i]
                else:
                    comp_color_i = comp_color
                    
                for j in range(n_comps_i):
                    if comp_color_i is None:
                        comp_color_i_j = comp_color_i
                    else:
                        comp_color_i_j = comp_color_i[j]
                    if comp_line:
                        ax.plot(di[x_energy], 
                            di['{} CPS{}_norm'.format(comp_id_i[j], bg_suffix)]*scale_i+si, 
                            linewidth=cls.linewidth*0.75, 
                            color=comp_color_i_j, 
                            zorder=100+i+1+j,
                            alpha=comp_line_alpha
                        )
                    if comp_fill:
                        ax.fill_between(di[x_energy], 
                            di['{} CPS{}_norm'.format(comp_id_i[j], bg_suffix)]*scale_i+si,
                            si,
                            color=comp_color_i_j,
                            alpha=comp_fill_alpha
                        )
            
            if plot_residuals:
                ax.plot(di[x_energy], di['residual_norm']*scale_i+si+residual_offset, 
                        linewidth=cls.linewidth*0.75, color=residual_color_i, zorder=100)

        # Aux.ax_opts(ax, major_tick_multiple=major_tick_multiple, 
        #             minor_tick_multiple=minor_tick_multiple, xlim=xlim, ylim=ylim)
        # ax.set_ylabel(ylabel, fontsize=cls.fontsize, labelpad=cls.labelpad*2/3)
        # ax.set_xlabel(xlabel, fontsize=cls.fontsize, labelpad=cls.labelpad)
        
        Plot.ax_opts(
            ax,
            # quantitative axis settings
            xlim=xlim,
            ylim=ylim,
            yticks=yticks, 
            xmajtm=major_tick_multiple, 
            xmintm=minor_tick_multiple,
            # line styling
            axes_linewidth=cls.axes_linewidth,
            tick_linewidth=cls.tick_linewidth,
            tick_length=cls.tick_length,
            # label settings
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=cls.fontsize,
            label_preset='xps',
            **kwargs
        )
        
        # if x_energy == 'B.E.':
        #     ax.invert_xaxis()

        ax.tick_params(labelsize=cls.fontsize)
        ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length, which='major')
        ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length*0.5, which='minor')

        fig.set_size_inches(*dim)
        
        if isinstance(top_text, str):
            ax.text(
                0.025, 
                1-0.04, 
                top_text, 
                horizontalalignment='left', 
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=cls.fontsize,
                weight='bold'
            )
            
        if legend:
            line = Line2D([0], [0], label='Fit', color='k', linewidth=cls.linewidth)
            point = Line2D([0], [0], label='Data', marker=marker, markersize=cls.marker_size, 
                    markeredgecolor='gray', markerfacecolor='gray', linestyle='', alpha=marker_alpha)
            handles = [line,point]

            if legend_fontsize is None:
                legend_fontsize = cls.fontsize

            ax.legend(handles=handles, 
                loc=legend_loc,
                frameon=False, 
                fontsize=legend_fontsize, 
                labelspacing=0.075/2, 
                borderpad=0, 
                handlelength=1, 
                handletextpad=0.2
            )

            
        if savefig:
                fig.savefig(savefig)
        return fig, ax
    
    # def plot_mesh(self,
    #     data,
    #     colormap,
    #     xlim,
    #     ylim
    # ):
    #     pass
    
    
    @staticmethod
    def findmaxcps(
        df, 
        xmin, 
        xmax, 
        col='CPS_no_bg_norm', 
        energy='B.E.', 
        xinterp_step=None,
        window_length=None,
        polyorder=3,
        **kwargs
    ):
        df1 = df[(df[energy] >= xmin) & (df[energy] <= xmax)].reset_index()
        df1 = df1.sort_values(by=energy)
        if xinterp_step:
            xinterp = np.arange(xmin, xmax, xinterp_step)
            cs = CubicSpline(df1[energy], df1[col])
            df1 = pd.DataFrame({energy: xinterp, col: cs(xinterp)})
        elif window_length:
            df1[col] = savgol_filter(df1[col], window_length, polyorder, **kwargs)
        idx = df1[col].argmax()
        return idx, df1[energy][idx]
    
    
    @staticmethod
    def interpolate(
        df,
        x,
        xcol='B.E.',
        ycol='CPS_no_bg_norm',
        method='makima',
        **kwargs
    ):
        idx = df.columns.get_loc(ycol)-1
        return Akima1DInterpolator(x, df.sort_values(by=xcol).drop(xcol, axis=1), method=method, **kwargs), idx


    @staticmethod
    def interpolate_and_average(
        df_list,
        x,
        xcol='B.E.',
        ycol='CPS_no_bg_norm',
        **kwargs
    ):
        xy_list = []
        for df in df_list:
            xy = np.empty((len(x), 2))
            cs, idx = interpolate(
                df,
                x,
                xcol=xcol,
                ycol=ycol,
                **kwargs
            )
            xy[:,0] = x
            xy[:,1] = cs(x)[:,idx]
            xy_list.append(xy)
        xy_avg = np.average(xy_list, axis=0)
        xy_std = np.std(xy_list, axis=0)
        return xy_avg, xy_std, xy_list