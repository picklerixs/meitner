import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from vamas import Vamas
from scipy.special import wofz
from lmfit import minimize, Parameters


class Pes:
    
    # fitting options
    background = 'shirley'
    params = {}
    
    # initialize Pes instance by taking dictionary of PES dataframes
    # df cols: be, ke, cps
    def __init__(self, df_dict, n_peaks=1):
        self.df_dict = df_dict
        self.keys_list = list(df_dict.keys())
        self.set_n_peaks(n_peaks)
        
    def plot_survey(self, keys_list=None, **kwargs):
        if keys_list == None:
            keys_list = self.keys_list
        for key in keys_list:
            self.survey_fig, self.survey_ax = plt.subplots()
            sns.scatterplot(data=self.df_dict[key], x='be', y='cps', **kwargs)
        
    # alternate constructor to import PES dataframes from VAMAS data
    # TODO: add automatic matching of partial region_id (e.g., C = C1 1 or S 2p = S2p/Se3p)
    @classmethod
    def from_vamas(cls, path, region_id=None, be_range=None, read_phi=False, shift=None, 
                     dict_keys=None, n_peaks=1, **kwargs):
        df_dict = {}
        
        # if single path is specified, repack into list
        if not cls.is_list_or_tuple(path):
            path = [path]
        
        for k in range(len(path)):
            
            region_id_k = cls.return_entry(region_id, k)
                
            if be_range == None or (be_range == False):
                be_range_k = (-np.inf,np.inf)
            elif cls.is_list_or_tuple(be_range[0]):
                be_range_k = be_range[k]
            elif cls.is_float_or_int(be_range[0]):
                be_range_k = be_range
            else:
                be_range_k = (-np.inf,np.inf)
            
            shift_k = cls.return_entry(shift, k)
            if shift_k == None or (shift_k == False):
                shift_k = 0
                
            read_phi_k = cls.return_entry(read_phi, k)
            
            key = cls.return_entry(dict_keys, k)
            if key == None:
                key = str(k)
                
            df = cls.import_single_vamas(path[k], region_id=region_id_k, shift=shift_k,
                                    be_range=be_range_k, read_phi=read_phi_k, **kwargs)
            df_dict.update({key: df})
        return cls(df_dict, n_peaks=n_peaks)
    
    # wrapper for Vamas() to extract a single PES dataframe from VAMAS data
    @staticmethod
    def import_single_vamas(path, region_id=None,
                # data processing options
                shift=None,
                be_range=None,
                read_phi=False,
                normalize=False):


        if path[-4:] == ".vms":
            # pull VAMAS data
            data = Vamas(path)
            
            # check spectra contained in VAMAS by pulling block_identifier for each block
            ids = [data.blocks[k].block_identifier for k in range(len(data.blocks))]
            
            # if spectrum was not specified, prompt user to select
            if region_id == None or (region_id == False):
                print('Found ' + str(len(ids)) + ' blocks with names')
                print(ids)
                region_id = input('Specify spectrum ID to access...')
                print()
                print(region_id)
            
            # get block index of desired spectrum
            if region_id in ids:
                idx = ids.index(region_id)
                
            # access spectrum, pull counts (c), generate KE range, and calculate BE from KE
            dataBlock = data.blocks[idx]
            cps = dataBlock.corresponding_variables[0].y_values # assumes counts are always the first corresponding variable...
            n = len(cps)
            ke = np.linspace(dataBlock.x_start, dataBlock.x_start + dataBlock.x_step*(n-1), n)
            
            # analyzer work function
            phi = dataBlock.analyzer_work_function_or_acceptance_energy
            # excitation energy
            hv = dataBlock.analysis_source_characteristic_energy
            print("Excitation energy:")
            print(str(hv) + " eV")
        else:
            warnings.warn(
                "Data format is not VAMAS! Aborting :C"
            )
            return

        # calculate binding energy
        if read_phi == None or (read_phi == False):
            phi = 0
            
        # shift kinetic energy to compensate for charging and/or hv error
        ke = ke - shift
        # compute binding energy after shift correction
        be = hv - ke - phi
        
        # note: trimming is done AFTER shift correction!
        trim = np.logical_and(be >= min(be_range), be <= max(be_range))
        ke = np.trim_zeros(ke*trim)
        be = np.trim_zeros(be*trim)
        cps = np.trim_zeros(cps*trim)
        
        if normalize:
            cps = cps/(max(cps)-min(cps))
        
        return pd.DataFrame(data={"ke": ke, "be": be, "cps": cps})
    
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
    
    # lineshapes
    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        return (amplitude/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-center)/sigma)**2)

    @staticmethod
    def lorentzian(x, amplitude, center, sigma):
        return amplitude*sigma/(np.pi*((x-center)**2+sigma**2))

    @classmethod
    def pseudo_voigt(cls, x, amplitude, center, sigma, fraction):
        return (1-fraction)*cls.gaussian(x,amplitude,center,sigma/np.sqrt(2*np.log(2))) + fraction*cls.lorentzian(x,amplitude,center,sigma)

    @staticmethod
    # borrowed from Andreas Seibert
    def voigt(x, amplitude, center, sigma, gamma):
        voigtfunction = np.sqrt(np.log(2))/(sigma*np.sqrt(np.pi)) * wofz((x-center)/sigma * np.sqrt(np.log(2)) + 1j * gamma/sigma * np.sqrt(np.log(2))).real
        return voigtfunction*amplitude
    
    @staticmethod
    # borrowed from Andreas Seibert
    def calc_voigt_fwhm(sigma, gamma):
        fl = 2*gamma
        fg = 2*sigma
        return fl/2 + np.sqrt(np.power(fl,2)/4 + np.power(fg,2))
    
    @staticmethod
    def linear(x, slope, intercept):
        return slope*x + intercept
    
    def fit_data(self, lineshape="voigt", bgdata=None, **kwargs):
        if bgdata == None:
            for key in self.keys_list:
                self.df_dict[key]['bg'] = self.calculate_shirley_background(self.df_dict[key]['cps'], **kwargs)
        self.result = minimize(self.residual, self.params)
    
    def residual(self, params, keys_list=None, lineshape="voigt", bgdata=None, *args, **kwargs):
        residuals = np.array([])
        # if keys_list == None:
        #     keys_list = self.keys_list
        for key in self.keys_list:
            # print(keys_list)
            # TODO figure out why keys_list gets turned into Parameters object..
            x_key = np.array(self.df_dict[key]['be'])
            y_key = np.array(self.df_dict[key]['cps'])
            # if background was fitted separately, subtract it manually before computing residuals
            # if isinstance(bgdata, dict):
            #     y_key += -bgdata[key]
            y_key += -np.array(self.df_dict[key]['cps'])
            resid = (y_key - self.generate_model_single_spectrum_no_bg(params, key, x_key, **kwargs))/(len(x_key)*np.linalg.norm(x_key))
            residuals = np.append(residuals, resid)
        return residuals
    
    def generate_params(self, be_guess, keys_list=None,
                        # general parameters
                        lineshape="voigt",
                        peak_ratios=False, peak_spacings=False,
                        expr_constraints=None,
                        # voigt/pseudo-voigt specific parameters (will not apply to pure gaussian or lorentzian peaks)
                        align_peaks=None, match_fwhm=None, fwhm_max=3, fwhm_min=0,
                        glmix_guess=0.5, glmix_max=1, glmix_min=0, glmix_vary=True, match_glmix=False,
                        sigma_guess=1, sigma_max=np.inf, sigma_min=0, sigma_vary=True, match_sigma='within',
                        gamma_guess=1, gamma_max=np.inf, gamma_min=0, gamma_vary=True, match_gamma=False,
                        *args, **kwargs):
        '''
        match:
            False: all varied independently
            'within': peaks within a single spectrum are matched to p0 in that spectrum
            True: all matched to p0 in first spectrum
        match_glmix only applies to pseudo-voigt peaks. Use match_sigma=True and match_gamma=True to contrain
        glmix for voigt peaks.
        '''
        
        # TODO if number of peaks is different for each spectrum, allow user to specify which peaks get aligned
        
        if keys_list == None:
            keys_list = self.keys_list
        
        self.params = Parameters()
        
        # each entry in be_guess_dict contains a list of guess positions for each region
        # preferred input
        if isinstance(be_guess, dict):
            self.be_guess = be_guess
            # ! no need to subset because of how dicts work
            # subset
            # self.be_guess = {key: be_guess[key] for key in keys_list}
            # TODO add check to ensure dict keys match
        # convert a single-valued input into expected dict format
        elif self.is_float_or_int(be_guess):
                self.be_guess = {key: [be_guess for _ in range(self.n_peaks[key])] for key in keys_list}
        elif self.is_list_or_tuple(be_guess):
            # convert a list of single-valued inputs into expected dict format
            if self.is_float_or_int(be_guess[0]):
                self.be_guess = {key: [be_guess[k] for k in range(self.n_peaks[key])] for key in keys_list}
            # convert a list of a list of guesses into dict format
            elif self.is_list_or_tuple(be_guess[0]):
                self.be_guess = dict(zip(keys_list, be_guess))
                
        for key in keys_list:
            x_key = np.array(self.df_dict[key]['be'])
            y_key = np.array(self.df_dict[key]['cps'])
                
            # set number of peaks for each dataframe
            n_peaks_key = self.n_peaks[key]
                
            for peak_number in range(n_peaks_key):
                peak_id = "data_{}_p{}_".format(key, peak_number)
                # basis parameters
                self.params.add(peak_id+"amplitude", value=(max(y_key)-min(y_key))*1.5, min=0)
                self.params.add(peak_id+"center", value=self.be_guess[key][peak_number], min=min(x_key), max=max(x_key))
                # for pseudo-Voigt, sigma is the width of both the Gaussian and the Lorentzian component
                # for Voigt, sigma is the width of the Gaussian component and gamma is of the Lorentzian component
                self.params.add(peak_id+"sigma", value=sigma_guess, min=sigma_min, max=sigma_max, vary=sigma_vary)
                # pseudo-Voigt-specific parameters
                if lineshape == "pseudo_voigt":
                    self.params.add(peak_id+"glmix", value=glmix_guess, min=glmix_min, max=glmix_max, vary=glmix_vary)
                    self.params.add(peak_id+"height", 
                            expr=peak_id+"amplitude/"+peak_id+"sigma*((1-"+peak_id+"glmix)/sqrt(pi/log(2))+1/(pi*"+peak_id+"sigma))")
                    self.params.add(peak_id+"fwhm", expr="2*"+peak_id+"sigma")
                # Voigt-specific parameters
                elif lineshape == "voigt":
                    self.params.add(peak_id+"gamma", value=gamma_guess, min=gamma_min, max=gamma_max)
                    self.params.add(peak_id+"gfwhm", expr="2*"+peak_id+"sigma")
                    self.params.add(peak_id+"lfwhm", expr="2*"+peak_id+"gamma")
                    self.params.add(peak_id+"glmix", expr=peak_id+"lfwhm/("+peak_id+"lfwhm+"+peak_id+"gfwhm)",
                            min=glmix_min, max=glmix_max)
                    self.params._asteval.symtable['calc_voigt_fwhm'] = self.calc_voigt_fwhm
                    self.params.add(peak_id+"fwhm", expr="calc_voigt_fwhm("+peak_id+"sigma,"+peak_id+"gamma)", 
                            min=fwhm_min, max=fwhm_max)
                
                for i in range(peak_number):
                    self.params.add("data_"+key+"_p"+str(peak_number)+"_p"+str(i)+"_ratio", expr=peak_id+"amplitude/data_"+key+"_p"+str(i)+"_amplitude", min=0, max=np.inf)
                    # setting min=0 ensures subsequent peaks increase in binding energy
                    self.params.add("data_"+key+"_p"+str(peak_number)+"_p"+str(i)+"_spacing", expr=peak_id+"center-data_"+key+"_p"+str(i)+"_center", min=0)
                
                if peak_number > 0:
                    if match_glmix == "within" and (lineshape == "pseudo_voigt"):
                        self.params.add(peak_id+"glmix", expr="data_"+key+"_p0_glmix")
                    if match_sigma == "within":
                        self.params.add(peak_id+"sigma", expr="data_"+key+"_p0_sigma")
                    if match_gamma == "within":
                        self.params.add(peak_id+"gamma", expr="data_"+key+"_p0_gamma")
                
                if ((peak_number > 0) or (key != keys_list[0])):
                    if lineshape == "pseudo_voigt": 
                        if ((match_glmix == "all") or (match_glmix == True)):
                            self.params.add(peak_id+"glmix", expr="data_"+keys_list[0]+"_p0_glmix")
                        if (match_fwhm == "all" or match_fwhm == True):
                            self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p0_sigma")
                            
                    if (match_sigma == "all" or match_sigma == True):
                        self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p0_sigma")
                    if (match_gamma == "all" or match_gamma == True):
                        self.params.add(peak_id+"gamma", expr="data_"+keys_list[0]+"_p0_gamma")
                        
                if key != keys_list[0]:
                    # constrain peak positions across different spectra?
                    # constrain all
                    if align_peaks == "all" or (align_peaks == True):
                        self.params.add(peak_id+"center", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_center")
                    # constrain only peaks specified in list
                    elif isinstance(align_peaks, list) or isinstance(align_peaks, tuple):
                        if peak_number == align_peaks[peak_number]:
                            self.params.add(peak_id+"center", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_center")
                    if lineshape == "pseudo_voigt":
                        if match_fwhm == "align":
                            self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_sigma")
                        if match_glmix == "align":
                            self.params.add(peak_id+"glmix", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_glmix")
                        
            # syntax: (i,j,k) where i = ID of peak 1, j = ID of peak 2, and k = value where i > j
            if peak_ratios != False and (isinstance(peak_ratios, list) or isinstance(peak_ratios, tuple)):
                dataKey = "data_"+key+"_"
                if isinstance(peak_ratios[0], int):
                    peakRatios = [peak_ratios]
                    
                for i in range(len(peak_ratios)):
                    peakRatios = peak_ratios[i]
                        
                    peak1 = "p"+str(max(peakRatios[0:2]))
                    peak2 = "p"+str(min(peakRatios[0:2]))
                    
                    if isinstance(self.n_peaks,dict):
                        if max(peakRatios[0:2]) > (self.n_peaks[key]):
                            break
                    # peak1 = "p"+str(peakRatios[0])
                    # peak2 = "p"+str(peakRatios[1])
                    
                    # print(peak1)
                    # print(peak2)
                    
                    # if peakRatios[2] == "align":
                    #     if k == 0:
                    #         params.add(dataKey+peak1+"_"+peak2+"_ratio")
                    #     else:
                    #         params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+keys_list[0]+"_"+peak1+"_"+peak2+"_ratio")
                    # else:
                    if peakRatios[2] == "align":
                        if key == keys_list[0]:
                            self.params.add(dataKey+peak1+"_"+peak2+"_ratio")
                        else:
                            self.params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+keys_list[0]+"_"+peak1+"_"+peak2+"_ratio")
                    else:
                        self.params.add(dataKey+peak1+"_"+peak2+"_ratio", value=peakRatios[2])
                        self.params[dataKey+peak1+"_"+peak2+"_ratio"].vary = False
                    self.params.add(dataKey+peak1+"_amplitude", expr=dataKey+peak2+"_amplitude*"+dataKey+peak1+"_"+peak2+"_ratio")
                
            if peak_spacings != False and (isinstance(peak_spacings, list) or isinstance(peak_spacings, tuple)):
                # for i in range(len(constrainPeakSpacings)):
                #     peakSpacings = constrainPeakSpacings[i]
                for peakSpacings in peak_spacings:
                    # print(peakSpacings)
                        
                    peak1 = "p"+str(max(peakSpacings[0:2]))
                    peak2 = "p"+str(min(peakSpacings[0:2]))
                    
                    if isinstance(self.n_peaks,dict):
                        if max(peakSpacings[0:2]) > self.n_peaks[key]:
                            print(key)
                            print(peak1,peak2)
                            break
                    
                    self.params.add(dataKey+peak1+"_"+peak2+"_spacing", value=peakSpacings[2])
                    if len(peakSpacings) > 3:
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].min = peakSpacings[3]
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].max = peakSpacings[4]
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].vary = True
                    elif len(peakSpacings) == 3:
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].vary = False
                    self.params.add(dataKey+peak1+"_center", expr=dataKey+peak2+"_center+"+dataKey+peak1+"_"+peak2+"_spacing")
                    # print(params[dataKey+peak1+"_center"])
                    
        # unpack user-specified expression-based constraints
        if expr_constraints != False and isinstance(expr_constraints, dict):
            for key in list(expr_constraints.keys()):
                if isinstance(expr_constraints[key], str):
                    self.params.add(key, expr=expr_constraints[key])
                elif isinstance(expr_constraints[key], dict):
                    self.params.add(key, **expr_constraints[key])
            
        # print(params)
        return self.params
    
    def set_n_peaks(self, n_peaks):
        '''
        Ensures n_peaks is a dict.
        '''
        if isinstance(n_peaks, dict):
            self.n_peaks = n_peaks
            # TODO add check to ensure dict keys match
        # generate dict from list/tuple
        elif self.is_list_or_tuple(n_peaks):
            self.n_peaks = dict(zip(self.keys_list, n_peaks))
        # generate dict from float/int
        elif self.is_float_or_int(n_peaks):
            self.n_peaks = dict(zip(self.keys_list, [n_peaks for _ in range(len(self.keys_list))]))
    
    def generate_model_single_spectrum_no_bg(self, params, key, x, model=0, lineshape="voigt"):
        model = 0
        for k in range(self.n_peaks[key]):
            peak_id = "data_{}_p{}_".format(key, k)
            
            if isinstance(lineshape, str):
                lineshape_k = lineshape
            elif self.is_list_or_tuple(lineshape):
                lineshape_k = lineshape[k]
                
            if lineshape_k == "gaussian":
                model += self.gaussian(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
            elif lineshape_k == "lorentzian":
                model += self.lorentzian(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
            elif lineshape_k == "pseudo_voigt":
                model += self.pseudo_voigt(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"glmix"])
            elif lineshape_k == "voigt":
                model += self.voigt(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
        return model

    def plotResult(self, 
                lineshape="voigt", normalize=False, 
                plotResiduals=True, text=False,
                tight_layout=True, minorTickMultiple=1,
                colors=colors, compzList=False, xdim=3.25, ydim=3.25, fontsize=10,
                xlim=False,ylim=False,xticks=False,saveFig=False,ypad=0):
        
        # xticks = False
        yticks = False
        # xdim = 3.25
        # ydim = 2
        j = 0
        for key in list(self.df_dict.keys()):
            # fig, ax = plt.subplots()
            fig = plt.figure()
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[5,1])
            ax, axResiduals = gs.subplots(sharex=True,sharey=False)
            x =  self.df_dict[key][0,:]
            y =  self.df_dict[key][1,:]
            
            # xlim = (min(x)-0.1,max(x)+0.1)
            # ylim = (min(y)-1000000,max(y)+1000000)
            # xlim = False
            # ylim = False
            
            if isinstance(self.n_peaks,int):
                n = self.n_peaks
                nmin = 0
            elif isinstance(self.n_peaks,dict):
                n = self.n_peaks[key]
                nmin = self.n_peaks[min(self.n_peaks, key=self.n_peaks.get)]
            
            if self.background == "linear":
                y_bg = linear(x, self.result.params["data_"+key+"_bg_slope"], self.result.params["data_"+key+"_bg_intercept"])
            elif self.background == "shirley":
                y_bg = shirleyIterative(y, self.result.params["data_"+key+"_bg_k"], 10, 100, 1e-8)
            elif self.background == False and bgdata != False:
                y_bg = bgdata[key]
            
            if self.background == False and bgdata != False:
                y_fit = fullModel(self.result.params, key, x, y-y_bg, n, self.background, lineshape=lineshape)
            else:
                y_fit = fullModel(self.result.params, key, x, y, n, self.background, lineshape=lineshape)

            
            marker = "+"
            markerResiduals = marker
            alpha = 0.5
            size = 5
            markeredgewidth = 2/3
            markerz=0
            envelopez=1
            if isinstance(normalize, int) and (normalize != False):
                y0 = self.result.params["data_"+key+"_p"+str(normalize)+"_height"]
                ax.plot(x,(y-y_bg)/y0, marker=marker, c='k', alpha=alpha, zorder=markerz, ms=size, mew=markeredgewidth, linestyle="None")
                ax.plot(x,(y_fit-y_bg)/y0, 'k-', linewidth=envelopelinewidth, zorder=envelopez, linestyle="None")
                ymax = 1
                ymin = 0
            else:
                ax.plot(x,y-y_bg, marker=marker, c='k', alpha=alpha, zorder=markerz, ms=size,  mew=markeredgewidth, linestyle="None")
                ymax = max(y-y_bg)
                ymin = min(y-y_bg)
                if self.background == False and bgdata != False:
                    # without simultaneous background fitting, y_fit generated by fullModel() does not contain background
                    ax.plot(x,y_fit, 'k-', linewidth=envelopelinewidth, zorder=envelopez)
                else:
                    # if plotting y_fit with simultaneous background fitting, need to subtract off background
                    ax.plot(x,y_fit-y_bg, 'k-', linewidth=envelopelinewidth, zorder=envelopez)
            # plt.plot(x,y_fit)
            if compzList == False:
                compzList = [1+k for k in range(n)]
            for k in range(n):
                peakId = "data_"+key+"_p"+str(k)+"_"
                # print(peakId)
                if lineshape == "pseudoVoigt":
                    y_comp = pseudoVoigt(x, self.result.params[peakId+"amplitude"].value, self.result.params[peakId+"center"].value,
                                        self.result.params[peakId+"sigma"].value, self.result.params[peakId+"fraction"].value)
                elif lineshape == "voigt":
                    y_comp = voigt(x, self.result.params[peakId+"amplitude"].value, self.result.params[peakId+"center"].value,
                                        self.result.params[peakId+"sigma"].value, self.result.params[peakId+"gamma"].value)
                if isinstance(normalize, int) and (normalize != False):
                    ax.plot(x,y_comp/y0, color=colors[k], linewidth=linewidth)
                else:
                    ax.plot(x,y_comp, color=colors[k], linewidth=linewidth, zorder=compzList[k])
            # plotOpts(fig, ax, 
            #      fontsize, xlim, ylim, xticks, yticks, minorTickMultiple, xdim, ydim,
            #      tight_layout=tight_layout)
            axOpts(ax,xlim,ylim,xticks,False,minorTickMultiple)
            if ypad != False or (ypad != 0):
                ax.set_ylim([ymin-(ymax-ymin)*0.05,ymax*(1+ypad)])
            ax.set_xlabel("Binding Energy (eV)", fontsize=fontsize)
            ax.set_ylabel("Intensity", fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            if text != False:
                ax.text(0.85,0.85,text[j],fontsize=fontsize,transform = ax.transAxes, horizontalalignment='center', verticalalignment='center')
            if tight_layout:
                plt.tight_layout()
            j += 1
            
            # figResiduals, axResiduals = plt.subplots()
            residuals = y - y_fit
            if self.background == False and bgdata != False:
                residuals += -y_bg
            axResiduals.plot(x,residuals/np.std(residuals), ms=size/2, c="k", alpha=alpha, marker=markerResiduals, mew=markeredgewidth, linestyle="None")
            axResiduals.axhline(y=0, color='k', linestyle='--', linewidth=envelopelinewidth)
            axResiduals.set_xlabel("Binding Energy (eV)", fontsize=fontsize)
            axResiduals.set_ylabel("${\it R}/\it{\sigma_{R}}$", fontsize=fontsize)
            axResiduals.tick_params(axis='both', which='major', labelsize=fontsize)
            # plotOpts(figResiduals, axResiduals, 
            #     fontsize, xlim, (-3,3), xticks, False, minorTickMultiple, xdim, max((ydim/3,1)),
            #     tight_layout=tight_layout)
            axOpts(axResiduals,xlim,(-3,3),xticks,False,minorTickMultiple)
            ax.invert_xaxis()
            figOpts(fig, fontsize, xdim, ydim)
            if tight_layout:
                plt.tight_layout()

            print(saveFig)
            if isinstance(saveFig,dict):
                fig.savefig(saveFig[key]+".svg")
            if isinstance(saveFig,str):
                fig.savefig(saveFig+".svg")

    # stolen from PyARPES
    @staticmethod
    def calculate_shirley_background(
        xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=(5,5)
    ) -> np.ndarray:
        """Core routine for calculating a Shirley background on np.ndarray data."""
        background = np.copy(xps)
        cumulative_xps = np.cumsum(xps, axis=0)
        total_xps = np.sum(xps, axis=0)

        rel_error = np.inf

        i_left = np.mean(xps[:n_samples[0]], axis=0)
        i_right = np.mean(xps[-n_samples[1]:], axis=0)

        iter_count = 0

        k = i_left - i_right
        for iter_count in range(max_iters):
            cumulative_background = np.cumsum(background, axis=0)
            total_background = np.sum(background, axis=0)

            new_bkg = np.copy(background)

            for i in range(len(new_bkg)):
                new_bkg[i] = i_right + k * (
                    (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                    / (total_xps - total_background + 1e-5)
                )

            rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

            background = new_bkg

            if np.any(rel_error < eps):
                break

        if (iter_count + 1) == max_iters:
            warnings.warn(
                "Shirley background calculation did not converge "
                + "after {} steps with relative error {}!".format(max_iters, rel_error)
            )
            
        return background