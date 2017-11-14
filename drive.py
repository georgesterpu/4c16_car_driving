import sys
sys.path.append('/usr/lib/python3.6/site-packages/')

import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import lycon
from utils import parse_position, preprocess_image, find_completion

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 30
MIN_SPEED = 10

speed_limit = MAX_SPEED

lap_definition = None

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # steering_angle = float(data["steering_angle"])
        # throttle = float(data["throttle"])

        x, y, z = parse_position(data["Position"])

        if lap_definition is not None:
            completion = find_completion([x,y,z], lap_definition)
            sys.stderr.write("\rTrack position: {0:3.2f}%".format(completion * 100))

        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = preprocess_image(image)
            image = np.array([image])

            steering_angle = float(model.predict(image, batch_size=1))

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            # print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            lycon.save(path='{}.jpg'.format(image_filename), image=image)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--lap_data',
        type=str,
        default='',
        help='Path to lap data (required for progress).'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # load model
    model = load_model(args.model)

    if args.lap_data != '':
        try:
            lap_definition = np.load(args.lap_data)
        except:
            print("Failed to load " + args.lap_data + "; no progress reporting.")


    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("**** Recording images from this run to {}.".format(args.image_folder))
    else:
        print("Image recording not enabled on this run.")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
