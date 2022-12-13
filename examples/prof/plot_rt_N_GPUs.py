import argparse
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
import re

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_time_bar(ax, events, num_gpus, order='ms', rate=10**6):
    ax.set_ylabel(f'Time [{order}]')
    bottoms = {key: 0. for key in ['runtime_max']} if args.use_max else {key: 0. for key in ['runtime']}
    for event in events:
        for key in bottoms:
            yval = float(times[key][event]) / rate  # ns -> ms by default
            if event in all_events:
                ax.bar(num_gpus +'GPUs', yval, bottom=bottoms[key], label=event, color=colors_by_events[event])
                all_events.remove(event)
            else:
                ax.bar(num_gpus +'GPUs', yval, bottom=bottoms[key], color=colors_by_events[event])
            """
            if pickle_file == 0 and len(args.pickle_path) == 1:
                ax.bar(str(pickle_file+1)+'GPU', yval, bottom=bottoms[key], label=event, color=colors_by_events[event])
            elif pickle_file == 1:
                ax.bar(str(num_gpus)+'GPUs', yval, bottom=bottoms[key], label=event, color=colors_by_events[event])
            else:
                ax.bar(str(pickle_file+1)+'GPUs', yval, bottom=bottoms[key], color=colors_by_events[event])
            """
            bottoms[key] += yval
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_path', type=str, nargs='+')
    parser.add_argument('--fig-path', type=str, default='data/prof.png')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--events', type=str, default='all')
    parser.add_argument('--ignore_events', type=str, default=None)
    parser.add_argument('--use_max', action='store_true', default=False)
    args = parser.parse_args()

    fig, ax = plt.subplots()
    
    ax.set_title(args.title)

    all_events = []
    for pickle_path in args.pickle_path:
        df = pd.read_pickle(args.pickle_path[1])
        pickle_events = df.index if args.events == 'all' else args.events.split(',')
        for event in pickle_events:
            if event not in all_events:
                all_events.append(event)

    if args.ignore_events is not None:
        ignore_events = args.ignore_events.split(',')
        for ignore_event in ignore_events:
            if ignore_event in all_events:
                all_events.remove(ignore_event)
    
    colors_by_events = dict(zip(all_events, colors))


    for pickle_path in args.pickle_path:
        df = pd.read_pickle(pickle_path)
        events = df.index if args.events == 'all' else args.events.split(',')
        if args.ignore_events is not None:
            ignore_events = args.ignore_events.split(',')
            df = df.drop(index=ignore_events)


        times = df.to_dict()
        if args.use_max:
            events = [event for event in events if event in times['runtime_max']]
        else:
            events = [event for event in events if event in times['runtime']]
        
        num_gpus = re.findall(r'\d+', pickle_path)[0]
        plot_time_bar(ax, events, num_gpus)

    plt.tight_layout()
    #plt.savefig(args.fig_path)
    plt.show()
