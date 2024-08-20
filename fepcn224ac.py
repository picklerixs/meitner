import meitner
import matplotlib.pyplot as plt

def main():
    stem = 'fepcn224ac/fepcn224-fe2p000'
    color = ['#672E45', '#E04E39']
    data = []
    for i in [1, 2]:
        path = '{}{}.csv'.format(stem, i)
        meitner.Casa.font_family = 'arial'
        df = meitner.Casa.load_csv(
            path
        )
        data.append(df)
        print(df)
    meitner.Casa.plot_stack(
        data,
        color,
        [703, 742],
        [-0.05, 0.39],
        comp_id=0,
        plot_comps=True,
        plot_envelope=True,
        subtract_bg=True,
        shift=0.185,
        ylabel='Normalized Intensity',
        dim=[5.5, 4.5],
        major_tick_multiple=10,
        minor_tick_multiple=2,
        savefig='fepcn224-fe2p-nocomps.svg'
    )
    plt.show()

if __name__ == '__main__':
    main()