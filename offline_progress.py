import argparse
import sys
import numpy as np

from utils import find_completion

def main():
    args = get_args()
    lap_data = load_lap_data(args.lap_data)
    track = load_car_positions()
    print("Loaded {} positions.".format(track.shape[0]))
    c = get_max_completion(track, lap_data)
    print(c)

def get_max_completion(t, l):
    last_completion = -1
    total_laps = 0
    for i in range(t.shape[0]):
        point = t[i, :]
        new_completion = find_completion(point, l)
        if (last_completion > 0.95 and new_completion < 0.5):
            total_laps += 1
        new_completion += total_laps
        if (new_completion > last_completion):
            last_completion = new_completion

    return last_completion


def get_args():
    parser = argparse.ArgumentParser(description='Offline Progress')
    parser.add_argument(
        '--lap_data',
        type=str,
        default='',
        help='Path to lap data (required).'
    )
    return parser.parse_args()


def load_lap_data(data_path):
    if (data_path == ''):
        print("Lap definition path must be supplied (--lap_data); can't continue!")
        sys.exit(-1)

    lap_definition = None
    try:
        lap_definition = np.load(data_path)
    except:
        print("Failed to load lap definition (path: '" + data_path + "'); can't continue!")
        sys.exit(-1)

    return lap_definition


def load_car_positions():
    D = np.load('car_positions.npz')
    return D['recorded_points']

if __name__ == '__main__':
    main()
