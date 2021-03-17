import numpy as np
import torch
import cv2
from typing import Tuple

IM_HEIGHT = 28
IM_WIDTH = 28
PX_MAX_VAL = 255

def load_data_from_csv(
    file: str,
    im_dims: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)

    # n = num samples, m = num features / sample
    n = np.shape(data)[0] 
    m = np.shape(data)[1] - 1

    # initialize empty arrays for x and y data
    x_data = np.zeros(shape=(n, im_dims[0], im_dims[1]))
    y_data = np.zeros(shape=(n, 1))

    for i in range(0, n):
        x = data[i][1:]
        img = np.reshape(x, im_dims)
        x_data[i] = img

        y_data[i] = data[i][0]
    
    return x_data, y_data


def preprocess_img(
    img: np.ndarray
) -> np.ndarray:
    """
    Takes an input greyscale image with pixels ranging from 0 to 255. Filters
    the image using Sobel filters and both x and y direction, and averages
    the result (corners high intensity, edges medium intensity, everything else
    low intensity). Normalizes the result's pixels to be float16 values ranging
    from 0 to 1

    Args:
        img (np.ndarray): input greyscale image. Must be a matrix (2 dimensions)
            and have pixel values somewhere in range 0-255.

    Returns:
        np.ndarray: result of applying sobel filters in both x and y directions,
            and normalizing all pixel values to range 0.0 - 1.0
    """
    res_x = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    res_y = cv2.Sobel(img, cv2.CV_8U, 0, 1)
    res = cv2.addWeighted(res_x, 0.5, res_y, 0.5, 0)
    res = res.astype(np.float32) / PX_MAX_VAL
    return res
