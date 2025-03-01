import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
from pycuda.compiler import SourceModule
import os
import numba as nb
import numpy as np


# Gets the image from the path specified by the user
def load_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if(img.ndim == 3):
        image_height, image_width, _ = img.shape
    
    elif(img.ndim == 4):
        image_height, image_width, _, _ = img.shape
    
    elif(img.ndim == 2):
        image_height, image_width = img.shape
    
    return img, image_height, image_width


def get_cuda_kernels_src_file():
    kernel_path  = os.path.join(os.getcwd(), "kernels", "seam_carving_kernels.cu")
    with open(kernel_path, "r") as f:
        kernels = f.read()

    ker = SourceModule(kernels)

    return ker


@nb.njit(nb.int32[:](nb.int32, nb.float32[:, :]), cache=True)
def get_backward_seam_from_idx(backtrack_idx, energy):
    image_height = energy.shape[0]
    image_width = energy.shape[1]
    current_idx = backtrack_idx
    seam = np.zeros(image_height, dtype=np.int32)
    seam[image_height-1] = backtrack_idx

    def min_index(idx1, idx2, arr):
        return idx1 if arr[idx1] < arr[idx2] else idx2

    for i in range(image_height - 2, -1, -1):

        # leftmost pixel
        if(current_idx % image_width == 0):
            seam[i] = min_index(current_idx, current_idx + 1, energy[i, :])
        
        # rightmost pixel
        elif((current_idx + 1) % image_width == 0):
            seam[i] = min_index(current_idx, current_idx - 1, energy[i, :])

        else:
            temp = min_index(current_idx, current_idx - 1, energy[i, :])
            seam[i] = min_index(current_idx + 1, temp, energy[i, :])

    return seam

