import argparse
import os
import numba as nb
import numpy as np
import cv2 as cv
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import math
import time


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    
    default_image_path = os.path.join(
        os.getcwd(), "examples", "images", "joel-filipe-QwoNAhbmLLo-unsplash.jpg"
    )
    
    parser.add_argument("--img_path", type=str, default=default_image_path, help="Path to image")
    parser.add_argument("--resized_width", type=int, default=400, help="Resized image width")
    parser.add_argument("--resized_height", type=int, default=400, help="Resized image height")

    return parser.parse_args()

def min_index(idx1, idx2, arr):
    return idx1 if arr[idx1] < arr[idx2] else idx2


@nb.njit(nb.int32[:](nb.int32, nb.float32[:, :]), cache=True)
def get_backward_seam_from_idx(backtrack_idx, energy):
    """Finds the min energy seam given a starting idx."""
    image_height = energy.shape[0]
    image_width = energy.shape[1]
    current_idx = backtrack_idx
    seam = np.zeros(image_height, dtype=np.int32)
    seam[image_height-1] = backtrack_idx

    for i in range(image_height - 2, -1, -1):

        # leftmost pixel
        if (current_idx % image_width == 0):
            seam[i] = min_index(current_idx, current_idx + 1, energy[i, :])

        # rightmost pixel
        elif ((current_idx + 1) % image_width == 0):
            seam[i] = min_index(current_idx, current_idx - 1, energy[i, :])

        else:
            temp = min_index(current_idx, current_idx - 1, energy[i, :])
            seam[i] = min_index(current_idx + 1, temp, energy[i, :])

    return seam


# Gets the image from the path specified by the user
def load_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if (img.ndim == 3):
        image_height, image_width, _ = img.shape

    elif (img.ndim == 4):
        image_height, image_width, _, _ = img.shape

    elif (img.ndim == 2):
        image_height, image_width = img.shape

    return img, image_height, image_width


def get_cuda_kernels_src_file():
    """Reads the CUDA kernels from the source file and compiles it"""
    kernel_path = os.path.join(
        os.getcwd(),"kernels", "seam_carving_kernels.cu")
    
    with open(kernel_path, "r") as f:
        kernels = f.read()
    ker = SourceModule(kernels)

    return ker


def initialize_cuda_kernels():
    """Loads CUDA kernels from source file."""
    ker = get_cuda_kernels_src_file()
    
    return {
        "rgb_to_gray": ker.get_function("Rgb2GrayWithPadding"),
        "sobel_x": ker.get_function("SobelHorizontal"),
        "sobel_y": ker.get_function("SobelVertical"),
        "energy_map": ker.get_function("EnergyMapBackward"),
        "cumulative_energy": ker.get_function("cumulativeMapBackward"),
        "find_min": ker.get_function("findMinInThreadBlock"),
        "remove_seam": ker.get_function("removeVerticalSeamAndInsertPadding"),
    }


def allocate_memory(image_height, image_width, img):
    """Allocates and transfers memory between host and device."""
    gray_image = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)
    gray_image_new = np.zeros((image_height + 2, image_width + 1), dtype=np.uint8)
    
    # Defines host arrays
    buffers = {
        "gray_image": gray_image,
        "gray_image_new": gray_image_new,
        "sobel_x": np.zeros((image_height, image_width), dtype=np.float32),
        "sobel_y": np.zeros((image_height, image_width), dtype=np.float32),
        "energy_map": np.zeros((image_height, image_width), dtype=np.float32),
        "cumulative_map": np.zeros((image_height, image_width), dtype=np.float32),
        "dummy_output": np.zeros((image_height, image_width), dtype=np.int32),
        "min_row": np.zeros(image_width, dtype=np.float32),
        "seam_indices": np.zeros(image_height, dtype=np.int32),
        "min_indices": np.zeros(math.ceil(image_width / 1024), dtype=np.int32),
    }

    # Allocates GPU memory
    device_buffers = {key: cuda.mem_alloc(arr.nbytes) for key, arr in buffers.items()}
    device_buffers["image"] = cuda.mem_alloc(img.nbytes)

    # Transfers memory from host to device
    cuda.memcpy_htod(device_buffers["image"], img)
    for key, d_mem in device_buffers.items():
        if key != "image":
            cuda.memcpy_htod(d_mem, buffers[key])

    return buffers, device_buffers 


def run_kernels(kernels, device_buffers, image_width, image_height):
    """Runs CUDA kernels in sequence."""
    threadsPerBlock = (16, 16, 1)
    numBlocks = ((image_width + 2 + threadsPerBlock[0] - 1) // threadsPerBlock[0], 
                (image_height + 2 + threadsPerBlock[1] - 1) // threadsPerBlock[1], 1)

    # Converts RGB to Grayscale
    start_time = time.time()
    kernels["rgb_to_gray"](device_buffers["image"], device_buffers["gray_image"],
                           np.int32(image_width), np.int32(image_height),
                           block=threadsPerBlock, grid=numBlocks)
    cuda.Context.synchronize()
    print(f"Rgb2Gray executed in {time.time() - start_time:.4f}s")

    # Runs Sobel Filters
    for kernel_name, d_output in [("sobel_x", "sobel_x"), ("sobel_y", "sobel_y")]:
        start_time = time.time()
        kernels[kernel_name](device_buffers["gray_image"], device_buffers[d_output],
                             np.int32(image_width), np.int32(image_height),
                             block=threadsPerBlock, grid=numBlocks)
        cuda.Context.synchronize()
        print(f"{kernel_name} executed in {time.time() - start_time:.4f}s")

    # Computes Energy Map
    start_time = time.time()
    kernels["energy_map"](device_buffers["sobel_x"], device_buffers["sobel_y"], device_buffers["energy_map"],
                          np.int32(image_width), np.int32(image_height),
                          block=threadsPerBlock, grid=numBlocks)
    cuda.Context.synchronize()
    print(f"Energy Map executed in {time.time() - start_time:.4f}s")


def find_seam(kernels, buffers, device_buffers, image_width, image_height):
    """Finds and removes the minimum energy seam."""
    cuda.memcpy_dtoh(buffers["energy_map"], device_buffers["energy_map"])
    buffers["cumulative_map"][0, :] = buffers["energy_map"][0, :]
    cuda.memcpy_htod(device_buffers["cumulative_map"], buffers["cumulative_map"])
     
    # Gets the cumulative energy map
    start_time = time.time()
    kernels["cumulative_energy"](device_buffers["energy_map"], device_buffers["cumulative_map"],
                                 device_buffers["dummy_output"],
                                 np.int32(image_height), np.int32(image_width),
                                 block=(1024, 1, 1), grid=(math.ceil(image_width / 1024), 1),
                                 shared=image_height * 4)
    cuda.Context.synchronize()
    print(f"Cumulative Energy executed in {time.time() - start_time:.4f}s")

    cuda.memcpy_dtoh(buffers["cumulative_map"], device_buffers["cumulative_map"])

    # Finds backtracking index in the last row of the cumulative energy map
    min_idx = np.argmin(buffers["cumulative_map"][image_height - 1, :])
    seam = get_backward_seam_from_idx(min_idx, buffers["cumulative_map"].reshape(image_height, image_width))
    
    # Transfers seam to device
    cuda.memcpy_htod(device_buffers["seam_indices"], seam)


def remove_seam(kernels, device_buffers, image_width, image_height):
    """Removes the minimum energy seam."""
    start_time = time.time()
    kernels["remove_seam"](device_buffers["seam_indices"], device_buffers["gray_image"],
                           device_buffers["gray_image_new"], np.int32(image_width), np.int32(image_height),
                           block=(16, 16, 1), grid=((image_width + 15) // 16, (image_height + 15) // 16))
    cuda.Context.synchronize()
    print(f"Seam removal executed in {time.time() - start_time:.4f}s")

