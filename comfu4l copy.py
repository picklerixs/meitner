import meitner
import pathlib
import matplotlib.pyplot as plt

def main():
    stem = 'comfu4l/echem/'
    color = ['gray', '#672E45', '#E04E39']
    comp_color = ['#007681' for _ in range(10)]
    data = []
    path_list = [stem + 'cootfs1_0003.csv', stem + 'coechem1_0003.csv']
    for path in path_list:
        path = pathlib.Path(path)
        meitner.Casa.font_family = 'arial'
        df = meitner.Casa.load_csv(
            path
        )
        data.append(df)
        print(df)
    meitner.Casa.plot_stack(
        data,
        color,
        [771, 809],
        [-0.03, 0.265],
        plot_envelope=True,
        subtract_bg=True,
        shift=0.14,
        ylabel='Normalized Intensity',
        dim=[5.5, 4.5],
        major_tick_multiple=10,
        minor_tick_multiple=2,
        data_style='markers',
        data_color='gray',
        legend=True,
        plot_comps=True,
        comp_id=[0,1,2,3],
        comp_color=comp_color,
        savefig='comfu4l/echem/fig/co2p.svg'
    )
    
    data = []
    path_list = [stem + 'cootfs1_0002.csv', stem + 'coechem1_0002.csv']
    for path in path_list:
        path = pathlib.Path(path)
        meitner.Casa.font_family = 'arial'
        df = meitner.Casa.load_csv(
            path
        )
        data.append(df)
        print(df)
    meitner.Casa.plot_stack(
        data,
        color,
        [680.5, 695.5],
        [-0.14, 1.0],
        plot_envelope=False,
        subtract_bg=True,
        shift=0.48,
        ylabel='Normalized Intensity',
        dim=[5.5, 4.5],
        major_tick_multiple=5,
        minor_tick_multiple=1,
        data_style='line',
        data_color=color,
        legend=False,
        savefig='comfu4l/echem/fig/f1s.svg'
    )
    
    data = []
    path_list = [stem + 'cootfs1_0001.csv', stem + 'coechem1_0001.csv']
    for path in path_list:
        path = pathlib.Path(path)
        meitner.Casa.font_family = 'arial'
        df = meitner.Casa.load_csv(
            path
        )
        data.append(df)
        print(df)
    meitner.Casa.plot_stack(
        data,
        color,
        [162.5, 175.5],
        [-0.14, 1.0],
        plot_envelope=True,
        subtract_bg=True,
        shift=0.485,
        ylabel='Normalized Intensity',
        dim=[5.5, 4.5],
        major_tick_multiple=5,
        minor_tick_multiple=1,
        data_style='markers',
        data_color='gray',
        legend=True,
        plot_comps=True,
        comp_color=comp_color,
        savefig='comfu4l/echem/fig/s2p.svg'
    )
    plt.show()

if __name__ == '__main__':
    main()