import lycon
import numpy as np

def parse_position(position_str):

    if position_str[0] == '(':
        parsed = position_str.lstrip('(').rstrip(')').split(',')
    else:
        parsed = position_str.split()

    x, y, z = [np.float(id) for id in parsed]
    return x, y, z


def preprocess_image(image):
    IMAGE_WIDTH = 200
    IMAGE_HEIGHT = 66
    image = image[50:140, :, :]
    image = lycon.resize(image, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, interpolation=lycon.Interpolation.CUBIC)
    return image


def find_completion(point, arr=None):
    if arr is None:
        return None

    N, _ = np.shape(arr)
    tmp = arr - point
    tmp = tmp**2
    tmp = np.sum(tmp, axis=-1)
    idx = np.argmin(tmp)
    return idx/N
