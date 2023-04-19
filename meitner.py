import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from vamas import Vamas
from scipy.special import wofz
from lmfit import Parameters
class Pes:
    
    # fitting options
    n_peaks = 1
    background = 'shirley'
    params = {}
    
    # initialize Pes instance by taking dictionary of PES dataframes
    # df cols: be, ke, cps
    def __init__(self, df_dict):
        self.df_dict = df_dict
        self.keys_list = list(df_dict.keys())
        
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
                     dict_keys=None, **kwargs):
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
        return cls(df_dict)
    
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
    def linear(x, slope, intercept):
        return slope*x + intercept
    
    def generate_params(self, keys_list=None,
                        lineshape="voigt",
                        beGuess=False, constrainPeakRatios=False,
                        constrainPeakSpacings=False,
                        constrainGL=True, alignPeaks="all", constrainFwhm="all", constraints=False,
                        exprConstraints=False, maxFwhm=np.inf, maxGlmix=1, minGlmix=0,
                        uniformSigma="within", *args):
        
        if keys_list == None:
            keys_list = self.keys_list
        
        self.params = Parameters()

        for key in keys_list:
            x = self.df_dict[key]['be']
            y = self.df_dict[key]['cps']
                
            # peak parameters
            if isinstance(self.n_peaks,int):
                n = self.n_peaks
                nmin = 0
            elif isinstance(self.n_peaks,dict):
                n = self.n_peaks[key]
                nmin = self.n_peaks[min(self.n_peaks, key=self.n_peaks.get)]
            if isinstance(beGuess,dict):
                peakCenter = beGuess[key]
            else:
                peakCenter = beGuess
                
            for k in range(n):
                peakId = "data_"+key+"_p"+str(k)+"_"
                # basis parameters
                params.add(peakId+"amplitude", value=(max(y)-min(y))*1.5, min=0)
                params.add(peakId+"center", value=peakCenter[k], min=min(x), max=max(x))
                # for pseudo-Voigt, sigma is the width of both the Gaussian and the Lorentzian component
                # for Voigt, sigma is the width of the Gaussian component and gamma is of the Lorentzian component
                params.add(peakId+"sigma", value=1.5, min=0, max=5)
                # pseudo-Voigt-specific parameters
                if lineshape == "pseudoVoigt":
                    params.add(peakId+"fraction", value=0.5, min=0, max=1)
                    params.add(peakId+"height", 
                            expr=peakId+"amplitude/"+peakId+"sigma*((1-"+peakId+"fraction)/sqrt(pi/log(2))+1/(pi*"+peakId+"sigma))")
                    params.add(peakId+"fwhm", expr="2*"+peakId+"sigma")
                # Voigt-specific parameters
                elif lineshape == "voigt":
                    params.add(peakId+"gamma", value=1.5, min=0, max=5)
                    params.add(peakId+"gfwhm", expr="2*"+peakId+"sigma")
                    params.add(peakId+"lfwhm", expr="2*"+peakId+"gamma")
                    params.add(peakId+"glmix", expr=peakId+"lfwhm/("+peakId+"lfwhm+"+peakId+"gfwhm)",
                            min=minGlmix, max=maxGlmix)
                    params._asteval.symtable['voigtFwhm'] = voigtFwhm
                    params.add(peakId+"fwhm", expr="voigtFwhm("+peakId+"sigma,"+peakId+"gamma)", 
                            min=0, max=maxFwhm)
                
                for j in range(k):
                    params.add("data_"+key+"_p"+str(k)+"_p"+str(j)+"_ratio", expr=peakId+"amplitude/data_"+key+"_p"+str(j)+"_amplitude", min=0, max=np.inf)
                    params.add("data_"+key+"_p"+str(k)+"_p"+str(j)+"_spacing", expr=peakId+"center-data_"+key+"_p"+str(j)+"_center")
                
                # if k > 0 and (k < nmin):
                if k > 0:
                    if constrainGL == "within" and (lineshape == "pseudoVoigt"):
                        params.add(peakId+"fraction", expr="data_"+key+"_p0_fraction")
                    if uniformSigma == "within":
                        params.add(peakId+"sigma", expr="data_"+key+"_p0_sigma")
                
                if ((k > 0) or (key != dataKeys[0])) and (k < nmin):
                    if ((constrainGL == "all") or (constrainGL == True)) and (lineshape == "pseudoVoigt"):
                        params.add(peakId+"fraction", expr="data_"+dataKeys[0]+"_p0_fraction")
                    if constrainFwhm == "all" or (uniformSigma == "all"):
                        params.add(peakId+"sigma", expr="data_"+dataKeys[0]+"_p0_sigma")
                if key != dataKeys[0] and (k < nmin):
                    # constrain peak positions across different spectra?
                    # constrain all
                    if alignPeaks == "all" or (alignPeaks == True):
                        params.add(peakId+"center", expr="data_"+dataKeys[0]+"_p"+str(k)+"_center")
                    # constrain only peaks specified in list
                    elif isinstance(alignPeaks, list) or isinstance(alignPeaks, tuple):
                        if k == alignPeaks[k]:
                            params.add(peakId+"center", expr="data_"+dataKeys[0]+"_p"+str(k)+"_center")
                    if constrainFwhm == "align":
                        params.add(peakId+"sigma", expr="data_"+dataKeys[0]+"_p"+str(k)+"_sigma")
                    if constrainGL == "align" and (lineshape == "pseudoVoigt"):
                        params.add(peakId+"fraction", expr="data_"+dataKeys[0]+"_p"+str(k)+"_fraction")
                        
            # syntax: (i,j,k) where i = ID of peak 1, j = ID of peak 2, and k = value where i > j
            if constrainPeakRatios != False and (isinstance(constrainPeakRatios, list) or isinstance(constrainPeakRatios, tuple)):
                dataKey = "data_"+key+"_"
                if isinstance(constrainPeakRatios[0], int):
                    peakRatios = [constrainPeakRatios]
                    
                for i in range(len(constrainPeakRatios)):
                    peakRatios = constrainPeakRatios[i]
                        
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
                    #         params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+dataKeys[0]+"_"+peak1+"_"+peak2+"_ratio")
                    # else:
                    if peakRatios[2] == "align":
                        if key == dataKeys[0]:
                            params.add(dataKey+peak1+"_"+peak2+"_ratio")
                        else:
                            params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+dataKeys[0]+"_"+peak1+"_"+peak2+"_ratio")
                    else:
                        params.add(dataKey+peak1+"_"+peak2+"_ratio", value=peakRatios[2])
                        params[dataKey+peak1+"_"+peak2+"_ratio"].vary = False
                    params.add(dataKey+peak1+"_amplitude", expr=dataKey+peak2+"_amplitude*"+dataKey+peak1+"_"+peak2+"_ratio")
                
            if constrainPeakSpacings != False and (isinstance(constrainPeakSpacings, list) or isinstance(constrainPeakSpacings, tuple)):
                # for i in range(len(constrainPeakSpacings)):
                #     peakSpacings = constrainPeakSpacings[i]
                for peakSpacings in constrainPeakSpacings:
                    # print(peakSpacings)
                        
                    peak1 = "p"+str(max(peakSpacings[0:2]))
                    peak2 = "p"+str(min(peakSpacings[0:2]))
                    
                    if isinstance(self.n_peaks,dict):
                        if max(peakSpacings[0:2]) > self.n_peaks[key]:
                            print(key)
                            print(peak1,peak2)
                            break
                    
                    params.add(dataKey+peak1+"_"+peak2+"_spacing", value=peakSpacings[2])
                    if len(peakSpacings) > 3:
                        params[dataKey+peak1+"_"+peak2+"_spacing"].min = peakSpacings[3]
                        params[dataKey+peak1+"_"+peak2+"_spacing"].max = peakSpacings[4]
                        params[dataKey+peak1+"_"+peak2+"_spacing"].vary = True
                    elif len(peakSpacings) == 3:
                        params[dataKey+peak1+"_"+peak2+"_spacing"].vary = False
                    params.add(dataKey+peak1+"_center", expr=dataKey+peak2+"_center+"+dataKey+peak1+"_"+peak2+"_spacing")
                    # print(params[dataKey+peak1+"_center"])
                    
        
        # unpack additional user-specified constraints, if any
        if constraints != False and isinstance(constraints, dict):
            for key in list(constraints.keys()):
                # if only constraint value is specified, default behavior is to fix parameter value
                if isinstance(constraints[key], float) or isinstance(constraints[key], int):
                    params[key].value = constraints[key]
                    params[key].vary = False
                # elif isinstance(constraints[key], list) or isinstance(constraints[key], tuple):
                #     params.set(value=constraints[key][0])
                #     params[key].vary = constraints[key][1]
                    
        # unpack user-specified expression-based constraints
        if exprConstraints != False and isinstance(exprConstraints, dict):
            for key in list(exprConstraints.keys()):
                if isinstance(exprConstraints[key], str):
                    params.add(key, expr=exprConstraints[key])
                # elif isinstance(exprConstraints[key], list) or isinstance(exprConstraints[key], tuple):
                    # if len(exprConstraints[key]) == 3:
                    #     params.add(key, expr=exprConstraints[key][0], min=exprConstraints[key][1], max=exprConstraints[key][2])
                    # elif len(exprConstraints[key]) == 2:
                    #     params.add(key, min=exprConstraints[key][0], max=exprConstraints[key][1])
                elif isinstance(exprConstraints[key], dict):
                    params.add(key, **exprConstraints[key])
            
        # print(params)
        return params
    
    def generate_model_single_spectrum_no_bg(self, key, x, model=0, lineshape="voigt"):
        self.model = model
        if isinstance(self.n_peaks, dict):
            n_peaks = self.n_peaks[key]
        elif self.is_float_or_int(self.n_peaks):
            n_peaks = self.n_peaks
        for k in range(n_peaks):
            peak_id = "data_"+key+"_p"+str(k)+"_"
            
            if isinstance(lineshape, str):
                lineshape_k = lineshape
            elif self.is_list_or_tuple(lineshape):
                lineshape_k = lineshape[k]
                
            if lineshape_k == "gaussian":
                self.model += self.gaussian(x, self.params[peak_id+"amplitude"],
                                    self.params[peak_id+"center"],self.params[peak_id+"sigma"],self.params[peak_id+"gamma"])
            elif lineshape_k == "lorentzian":
                self.model += self.lorentzian(x, self.params[peak_id+"amplitude"],
                                    self.params[peak_id+"center"],self.params[peak_id+"sigma"],self.params[peak_id+"gamma"])
            elif lineshape_k == "pseudo_voigt":
                self.model += self.pseudo_voigt(x, self.params[peak_id+"amplitude"],
                                    self.params[peak_id+"center"],self.params[peak_id+"sigma"],self.params[peak_id+"fraction"])
            elif lineshape_k == "voigt":
                self.model += self.voigt(x, self.params[peak_id+"amplitude"],
                                    self.params[peak_id+"center"],self.params[peak_id+"sigma"],self.params[peak_id+"gamma"])

