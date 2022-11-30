import argparse
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_time_bar(ax, events, pickle_file, order='ms', rate=10**6):
    ax.set_ylabel(f'Time [{order}]')
    bottoms = {key: 0. for key in ['runtime_max']} if args.use_max else {key: 0. for key in ['runtime']}
    for event in events:
        for key in bottoms:
            yval = float(times[key][event]) / rate  # ns -> ms by default
            if pickle_file == 0 and len(args.pickle_path) == 1:
                ax.bar(str(pickle_file+1)+'GPU', yval, bottom=bottoms[key], label=event, color=colors_by_events[event])
            elif pickle_file == 1:
                ax.bar(str(pickle_file+1)+'GPUs', yval, bottom=bottoms[key], label=event, color=colors_by_events[event])
            else:
                ax.bar(str(pickle_file+1)+'GPUs', yval, bottom=bottoms[key], color=colors_by_events[event])
            bottoms[key] += yval
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_path', type=str, nargs='+')
    parser.add_argument('--fig-path', type=str, default='data/prof.png')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--events', type=str, default='all')
    parser.add_argument('--sub_events', type=str)
    parser.add_argument('--use_max', action='store_true', default=False)
    args = parser.parse_args()

    fig, ax = plt.subplots()
    
    ax.set_title(args.title)

    if len(args.pickle_path) > 1:   # first plot will always be 1GPU and the next ones multi GPUs (which also profile communications)
        df = pd.read_pickle(args.pickle_path[1])
        all_events = df.index if args.events == 'all' else args.events.split(',')
        colors_by_events = {}
        for i, event in enumerate(all_events):
            colors_by_events[event] = colors[i]
    else:
        df = pd.read_pickle(args.pickle_path[0])
        all_events = df.index if args.events == 'all' else args.events.split(',')
        colors_by_events = {}
        for i, event in enumerate(all_events):
            colors_by_events[event] = colors[i]


    for i, pickle_path in enumerate(args.pickle_path):
        df = pd.read_pickle(pickle_path)
        events = df.index if args.events == 'all' else args.events.split(',')
        times = df.to_dict()
        if args.use_max:
            events = [event for event in events if event in times['runtime_max']]
        else:
            events = [event for event in events if event in times['runtime']]
        
        plot_time_bar(ax, events, i)

    plt.tight_layout()
    #plt.savefig(args.fig_path)
    plt.show()
