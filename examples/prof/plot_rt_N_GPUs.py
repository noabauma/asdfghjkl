import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_time_bar(ax, order='ms', rate=10**6):
    ax.set_ylabel(f'Time [{order}]')
    bottoms = {key: 0. for key in ['runtime']}
    for i, event in enumerate(events):
        color = colors[i]
        for key in bottoms:
            yval = float(times[key][event]) / rate  # ns -> ms by default
            label = event if key == 'runtime' else None
            ax.bar(key, yval, bottom=bottoms[key], label=label, color=color)
            bottoms[key] += yval
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')


def main():
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(args.title)
    plot_time_bar(ax)
    plot_time_bar(ax)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_path', type=str, nargs='+')
    parser.add_argument('--fig-path', type=str, default='data/prof.png')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--events', type=str, default='all')
    parser.add_argument('--sub_events', type=str)
    args = parser.parse_args()


    df = pd.read_pickle(args.pickle_path[0])
    events = df.index if args.events == 'all' else args.events.split(',')
    times = df.to_dict()
    events = [event for event in events if event in times['runtime']]
    main()
