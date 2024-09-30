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
     