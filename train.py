import model
from os import path
import csv
import numpy as np
import lycon
from keras.models import save_model


def import_csv_data(logfile):
    with open(logfile, 'r') as f:
        data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    center = 0
    left = 1
    right = 2
    angle = 3
    accel = 4
    stop = 5
    speed = 6

    parsed_data = dict()

    for line in data:
        if float(line[speed]) < 0.01:
            continue  # skip examples with a quasi-static car

        parsed_data[line[center]] = float(line[angle])
        parsed_data[line[left]] = float(line[angle]) + 0.20
        parsed_data[line[right]] = float(line[angle]) - 0.20

    return parsed_data


def training_generator(training_data, batch_size):
    inputs_batch = []
    targets_batch = []

    while True:
        for idx, (k, v) in enumerate(training_data.items()):
            decoded_image = lycon.load(k)
            decoded_image = decoded_image[50:140, :, :]
            decoded_image = lycon.resize(decoded_image, width=200, height=66, interpolation=lycon.Interpolation.CUBIC)

            label = v

            inputs_batch.append(decoded_image)
            targets_batch.append(label)

            if (idx+1) % batch_size == 0:
                yield (np.asarray(inputs_batch), np.asarray(targets_batch))
                inputs_batch.clear()
                targets_batch.clear()


def main():

    recordings_dir = './recordings/'

    driving_log = path.join(recordings_dir, 'driving_log.csv')

    training_data = import_csv_data(driving_log)

    angles = []
    for k, v in training_data.items():
        angles.append(v)

    dnn_model = model.build_model()
    dnn_model.fit_generator(
        generator=training_generator(training_data, batch_size=64),
        steps_per_epoch=len(training_data)//64,
        epochs=10,
        verbose=1,
        shuffle=True
    )

    save_model(dnn_model, './model.h5')


if __name__ == '__main__':
    main()
