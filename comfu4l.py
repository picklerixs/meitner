import meitner
import matplotlib.pyplot as plt

def main():
    stem = 'comfu4l/co27otf000'
    color = ['gray', '#672E45', '#E04E39']
    data = []
    for i in [6, 2, 4]:
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
        [763, 809],
        [-0.07, 0.6],
        plot_comps=False,
        plot_envelope=False,
        subtract_bg=True,
        shift=0.185,
        ylabel='Normalized Intensity',
        dim=[5.5, 4.5],
        major_tick_multiple=10,
        minor_tick_multiple=2,
        data_style='line',
        data_color=color,
        savefig='comfu4l-co2p.svg'
    )
    plt.show()

if __name__ == '__main__':
    main()