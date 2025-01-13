import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import decimate, resample, savgol_filter
from scipy.interpolate import UnivariateSpline


class Exafs:
    def __init__(
        self,
        file=None,
        df=None,
        **kwargs
    ):
        '''
        KWARGS:
            df (pd.DataFrame): DataFrame to parse (df.columns = ['Energy (eV)', 'Intensity', 'Ref Intensity'], 'Ref Intensity' is optional)
            file (str): path to file (df is always preferred if both df and file are specified)
        df should have columns ['Energy (eV)', 'Intensity', 'Ref Intensity']
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
        *args,
        **kwargs
    ):
        df = pd.read_csv(
            file,
            sep=r"\s+",
            skiprows=13,
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
        self.df = df
        return self.df
    
    def rebin(
        self,
        E0,
        pre_edge_cutoff=-300,
        Emax=None,
        kmax=None,
        xanes_region=[-30, 50],
        pre_edge_step=10, # eV
        exafs_step=0.05, # 1/Å
        s=0
    ):
        self.df_orig = np.copy(self.df)
        # pre_edge = np.array(pre_edge) + E0
        # exafs = self.k_to_e(np.array(exafs)) + E0
        
        pre_edge = np.array([pre_edge_cutoff, xanes_region[0]]) + E0
        if Emax is None:
            Emax = self.df['Energy (eV)'].max() - E0
        elif Emax > (self.df['Energy (eV)'].max() - E0):
            Emax = self.df['Energy (eV)'].max() - E0
        print(Emax)
        if kmax is None:
            kmax = self.E_to_k(Emax, 0)
            print('kmax = {}'.format(kmax))
        # generate EXAFS grid in k-space
        exafs_grid = np.arange(0, kmax, exafs_step)
        # convert EXAFS grid to E-space
        exafs_grid = self.k_to_E(exafs_grid, E0)
        exafs_grid = exafs_grid[exafs_grid >= xanes_region[1] + E0]
        exafs_grid = exafs_grid[exafs_grid <= Emax + E0]
        
        # truncate below pre_edge[0]
        self.df[self.df['Energy (eV)'] >= pre_edge[0]]
        # truncate above exafs[1]
        self.df[self.df['Energy (eV)'] <= Emax]
        
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
        plot=False,
        save=True,
        **kwargs
    ):
        if plot:
            fig, ax = plt.subplots(layout='constrained')
        for i in range(len(file_list)):
            xafs = Exafs("{}{}".format(file_list[i], file_extension))
            xafs.rebin(
                E0,
                **kwargs
            )
            if save:
                xafs.to_csv("{}{}".format(out_names[i], out_extension))
            if plot:
                ax.plot(xafs.df['Energy (eV)'], xafs.df['Intensity'], 'ko')
        if plot:
            plt.show()