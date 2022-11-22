import argparse
import sqlite3
import warnings

import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_all_event_texts():
    sql = f"""
    SELECT DISTINCT text
    FROM NVTX_EVENTS;
    """
    df = pd.read_sql(sql, con)
    return [row['text'] for _, row in df.iterrows()]


def get_event_start_end(event_text):
    sql = f"""
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = '{event_text}';
    """
    df = pd.read_sql(sql, con)
    return [(row['start'], row['end']) for _, row in df.iterrows()]


def get_total_time_in_event(target_table_name, event_start, event_end):
    sql = f"""
    SELECT SUM(target.end - target.start) AS total_time
    FROM {target_table_name} target
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime
      ON target.correlationId = runtime.correlationId
    WHERE runtime.start BETWEEN {event_start} AND {event_end};
    """
    try:
        df = pd.read_sql(sql, con)
        time = df['total_time'].iloc[0]
        if time is None:
            return 0
        return time
    except:
        return 0
    

def main():
    if args.event_texts is None:
        event_texts = get_all_event_texts()
        if args.event_keywords is not None:
            event_keywords = args.event_keywords.split(',')
            event_texts = [txt for txt in event_texts
                           if any([kwd in txt for kwd in event_keywords])]
    else:
        event_texts = args.event_texts.split(',')
        if args.event_keywords is not None:
            warnings.warn('As event_texts is specified, event_keywords will be ignored.')

    warmup_start = 0

    times = dict()
    index = []
    #print(f'Collecting time for {event_texts}')
    for txt in event_texts:
        event_start_end = get_event_start_end(txt)

        if args.ignore_warmup and (txt == 'Warmup' or txt == 'warmup'):          #every nvtx event before warmup will be ignored
            warmup_start = event_start_end[0][1]
            continue

        if txt == 'init' or txt == 'Iter':
            continue

        Nones = any([s == None or e == None for s, e in event_start_end])
        if len(event_start_end) == 0 or Nones:
            continue
        index.append(txt)
        if args.ignore_first_event:
            # ignore first NVTX event
            event_start_end = event_start_end[1:]

        if warmup_start > 0:
            for enum, (s, e) in enumerate(event_start_end):
                if warmup_start < s:
                    break
            event_start_end = event_start_end[enum:]
        
        for i in range(len(event_start_end)):
            event_start_end[i] = (event_start_end[i][0]/1e6, (event_start_end[i][1] - event_start_end[i][0])/1e6)

        times[txt] = event_start_end

        #print("@@@", txt, ": ", event_start_end, "@@@")

    return times
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite_path', type=str, nargs='+')
    parser.add_argument('--pickle_path', type=str, default='data/nvtx_events.pickle')
    parser.add_argument('--ignore_first_event', action='store_true', default=False)
    parser.add_argument('--ignore_warmup', action='store_true', default=True)
    parser.add_argument('--event_texts', type=str)
    parser.add_argument('--event_keywords', type=str)
    parser.add_argument('--wandb_run_path', type=str, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots()

    ax.grid()

    for gpu, sq_ath in enumerate(args.sqlite_path):
        con = sqlite3.connect(sq_ath)
        times = main()

        num_keys = len(times)

        for enum, (key, array) in enumerate(times.items()):
            if gpu == 0:
                ax.broken_barh(array, ((gpu - 0.45), 0.9), label=key, color=colors[enum])
            else:
                ax.broken_barh(array, ((gpu - 0.45), 0.9), color=colors[enum])

    ax.set_xlabel('times [ms]')
    ax.set_ylabel('GPU') 

    ax.set_yticks([i for i in range(len(args.sqlite_path))])

    plt.legend(loc='best')

    plt.show()
