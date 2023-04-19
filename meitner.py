import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from vamas import Vamas

class Pes:
    
    # initialize Pes instance by taking dictionary of PES dataframes
    # df cols: be, ke, cps
    def __init__(self, df_dict):
        self.df_list = df_dict
        
    # alternate constructor to import PES dataframes from VAMAS data
    # TODO: add automatic matching of partial region_id (e.g., C = C1 1 or S 2p = S2p/Se3p)
    @classmethod
    def import_vamas(cls, path, region_id=None, be_range=None, read_phi=False, shift=None, 
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