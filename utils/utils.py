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
    
    parser.add_argument("--img_path", type=str, help="Path to image")
    parser.add_argument("--resized_width", type=int, help="Resized image width")
    parser.add_argument("--resized_height", type=int, help="Resized image height")

    return parser.parse_args()

@nb.njit(nb.int32[:](nb.int32, nb.float32[:, :], nb.int64, nb.int64), cache=True)
def get_backward_seam_from_idx(backtrack_idx, energy, height, width):
    """Finds the min energy seam given a starting idx."""
    image_height = height
    image_width = width
    current_idx = backtrack_idx
    seam = np.zeros(image_height, dtype=np.int32)
    seam[image_height-1] = backtrack_idx
    
    # Helper function
    def min_index(idx1, idx2, arr):
        return idx1 if arr[idx1] < arr[idx2] else idx2

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
        
        current_idx = seam[i]

    return seam


# Gets the image from the path specified by the user
def load_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    if (img.ndim == 3):
        image_height, image_width, _ = img.shape
    elif (img.ndim == 2):
        image_height, image_width = img.shape
    else:
        raise ValueError(f"Expected image dimension to be 2 or 3, got {img.ndim}")
    
    return img, image_height, image_width


def save_image(image, image_path):
    """Takes in an image and saves it"""
    suffix = image_path.split('\\')[-1]
    suffix = "Resized_" + suffix
    save_path = os.path.join(os.getcwd(), "examples", "results", suffix)
    cv.imwrite(save_path, image)


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
        "update_energy_map": ker.get_function("updateEnergyMap"),
        "remove_seam_RGB" : ker.get_function("removeSeamRGB"),
        "remove_seam_gray" : ker.get_function("removeVerticalSeam"),
        "remove_seam_maps" : ker.get_function("removeVerticalSeamMaps")
    }


def allocate_memory(image_height, image_width, img):
    """Allocates and transfers memory between host and device."""
    gray_image = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)
    gray_image_new = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)
    
    if (img.ndim == 3):
        R = np.ascontiguousarray(img[:, :, 0])
        G = np.ascontiguousarray(img[:, :, 1])
        B = np.ascontiguousarray(img[:, :, 2])
        
        # Defines host arrays
        buffers = {
            "gray_image": gray_image,
            "gray_image_new": gray_image_new,
            "R": R,
            "G": G,
            "B": B,
            "R_new": np.zeros((image_height, image_width), dtype=np.uint8),
            "G_new": np.zeros((image_height, image_width), dtype=np.uint8),
            "B_new": np.zeros((image_height, image_width), dtype=np.uint8),
            "sobel_x": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_x_new": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_y": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_y_new": np.zeros((image_height, image_width), dtype=np.float32),
            "energy_map": np.zeros((image_height, image_width), dtype=np.float32),
            "energy_map_new": np.zeros((image_height, image_width), dtype=np.float32),
            "cumulative_map": np.zeros((image_height, image_width), dtype=np.float32),
            "min_row": np.zeros(image_width, dtype=np.float32),
            "seam_indices": np.zeros(image_height + 2, dtype=np.int32),
            "min_indices": np.zeros(math.ceil(image_width / 1024), dtype=np.int32),
    }

    elif (img.ndim == 2):
            # Defines host arrays
        buffers = {
            "gray_image": gray_image,
            "gray_image_new": gray_image_new,
            "sobel_x": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_x_new": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_x": np.zeros((image_height, image_width), dtype=np.float32),
            "sobel_y_new": np.zeros((image_height, image_width), dtype=np.float32),
            "energy_map": np.zeros((image_height, image_width), dtype=np.float32),
            "energy_map_new": np.zeros((image_height, image_width), dtype=np.float32),
            "cumulative_map": np.zeros((image_height, image_width), dtype=np.float32),
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
        if key != "image" and key != "seam_indices":
            cuda.memcpy_htod(d_mem, buffers[key])

    return buffers, device_buffers 


def run_kernels(kernels, device_buffers, image_width, image_height):
    """Runs CUDA kernels in sequence."""
    threadsPerBlock = (32, 32, 1)
    numBlocks = ((image_width + 2 + threadsPerBlock[0] - 1) // threadsPerBlock[0], 
                (image_height + 2 + threadsPerBlock[1] - 1) // threadsPerBlock[1], 1)

    # Converts RGB to Grayscale
    start_time = time.time()
    kernels["rgb_to_gray"](device_buffers["image"], device_buffers["gray_image"],
                           np.int32(image_width), np.int32(image_height),
                           block=threadsPerBlock, grid=numBlocks)
    cuda.Context.synchronize()
    # print(f"Rgb2Gray executed in {time.time() - start_time:.4f}s")

    # Runs Sobel Filters
    for kernel_name, d_output in [("sobel_x", "sobel_x"), ("sobel_y", "sobel_y")]:
        start_time = time.time()
        kernels[kernel_name](device_buffers["gray_image"], device_buffers[d_output],
                             np.int32(image_width), np.int32(image_height),
                             block=threadsPerBlock, grid=numBlocks)
        cuda.Context.synchronize()
        # print(f"{kernel_name} executed in {time.time() - start_time:.4f}s")

    # Computes Energy Map
    start_time = time.time()
    kernels["energy_map"](device_buffers["sobel_x"], device_buffers["sobel_y"], device_buffers["energy_map"],
                          np.int32(image_width), np.int32(image_height),
                          block=threadsPerBlock, grid=numBlocks)
    cuda.Context.synchronize()
    # print(f"Energy Map executed in {time.time() - start_time:.4f}s")


def find_seam(kernels, buffers, seam, device_buffers, image_width, image_height):
    """Finds and removes the minimum energy seam."""
    cuda.memcpy_dtoh(buffers["energy_map"], device_buffers["energy_map"])
    buffers["cumulative_map"][0, :image_width] = buffers["energy_map"][0, :image_width]
    cuda.memcpy_htod(device_buffers["cumulative_map"], buffers["cumulative_map"])

    # Gets the cumulative energy map
    start_time = time.time()
    kernels["cumulative_energy"](device_buffers["energy_map"], device_buffers["cumulative_map"],
                                 np.int32(image_height), np.int32(image_width),
                                 block=(1024, 1, 1), grid=(1, 1),
                                 shared=image_height * 4)
    cuda.Context.synchronize()
    # print(f"Cumulative Energy kernel executed in {time.time() - start_time:.4f}s")
    
    cumulative_map_updated = np.zeros((image_height, image_width), dtype=np.float32)
    cuda.memcpy_dtoh(cumulative_map_updated, device_buffers["cumulative_map"])
   
    # defines index of the last active row of cumulative map(since size decreases)
    last_active_row_idx = (image_height - 1) * image_width
    buffers["min_row"] = cumulative_map_updated.ravel()[last_active_row_idx : last_active_row_idx + image_width]

    cuda.memcpy_htod(device_buffers["min_row"], buffers["min_row"])

    # Finds backtracking index in the last row of the cumulative energy map
    start_time = time.time()
    kernels["find_min"](device_buffers["min_row"], device_buffers["min_indices"],
                         np.int32(image_width),
                         block=(1024, 1, 1), grid=(math.ceil(image_width / 1024), 1),)
    cuda.Context.synchronize()
    # print(f"Find min kernel executed in {time.time() - start_time:.4f}s")

    cuda.memcpy_dtoh(buffers["min_indices"], device_buffers["min_indices"])
    
    # Checks for bounds on min_indices
    accept_map = buffers["min_indices"] < image_width

    # Gets the backtracking start index
    min_idx = buffers["min_indices"][np.argmin(buffers["min_row"][buffers["min_indices"][accept_map]])]

    # Seam indices of the shape image_height + 2 to make removing seam while preserving padding easier
    seam[1: image_height + 1] = get_backward_seam_from_idx(min_idx, cumulative_map_updated, image_height, image_width)
    seam[0] = seam[1]
    seam[image_height + 1] = seam[image_height]

    # Transfers seam to device
    cuda.memcpy_htod(device_buffers["seam_indices"], seam)


def remove_seam(kernels, device_buffers, image_width, image_height):
    """Removes the minimum energy seam."""
    
    # Removes the vertical seam from the gray image
    kernels["remove_seam_gray"](device_buffers["seam_indices"], device_buffers["gray_image"],
                               device_buffers["gray_image_new"], np.int32(image_width), np.int32(image_height),
                               block=(1024, 1, 1), grid=(math.ceil((image_height + 2) * (image_width +  1) / 1024), 1)
                            )
    cuda.Context.synchronize()
    # print(f"Seam removal executed in {time.time() - start_time:.4f}s")
    
    # Swaps buffers to make sure that gray_image is always up to date
    device_buffers["gray_image"], device_buffers["gray_image_new"] =  (
        device_buffers["gray_image_new"], device_buffers["gray_image"]
    )
    
    # Removes the vertical seam from energy and sobel maps
    kernels["remove_seam_maps"](device_buffers["seam_indices"], device_buffers["energy_map"], device_buffers["energy_map_new"],
                               device_buffers["sobel_x"], device_buffers["sobel_x_new"],
                               device_buffers["sobel_y"], device_buffers["sobel_y_new"], 
                               np.int32(image_width), np.int32(image_height),
                               block=(1024, 1, 1), grid=(math.ceil(image_height * (image_width -  1) / 1024), 1)
                            )
    cuda.Context.synchronize()
    
    # Swaps buffers to make sure that maps are always up to date
    device_buffers["energy_map"], device_buffers["energy_map_new"] =  (
        device_buffers["energy_map_new"], device_buffers["energy_map"]
    )

    device_buffers["sobel_x"], device_buffers["sobel_x_new"] =  (
        device_buffers["sobel_x_new"], device_buffers["sobel_x"]
    )

    device_buffers["sobel_y"], device_buffers["sobel_y_new"] =  (
        device_buffers["sobel_y_new"], device_buffers["sobel_y"]
    )

def update_energy_map(kernels, device_buffers, image_width, image_height):
    """Updates the energy map after removal of a seam"""
    start_time = time.time()
    kernels["update_energy_map"](device_buffers["seam_indices"], device_buffers["gray_image"],
                           device_buffers["sobel_x"], device_buffers["sobel_y"], device_buffers["energy_map"], 
                           np.int32(image_width), np.int32(image_height),
                           block=(3, 32, 1), grid=((image_height + 31) // 32, 1))
    cuda.Context.synchronize()
    # print(f"Update energy map executed in {time.time() - start_time:.4f}s")

def remove_seam_from_RGB(kernels, device_buffers, image_width, image_height):
    start_time = time.time()
    kernels["remove_seam_RGB"](
        device_buffers["R"], device_buffers["G"], device_buffers["B"],
        device_buffers["R_new"], device_buffers["G_new"], device_buffers["B_new"],
        device_buffers["seam_indices"], np.int32(image_width), np.int32(image_height),
        block=(1024, 1, 1), grid=(math.ceil(image_height * (image_width -  1) / 1024), 1)
    )
    cuda.Context.synchronize()
    # print(f"Remove seam from RGB executed in {time.time() - start_time:.4f}s")

    # Swaps buffers 
    device_buffers["R"], device_buffers["R_new"] = (
        device_buffers["R_new"], device_buffers["R"]
    )

    device_buffers["G"], device_buffers["G_new"] = (
        device_buffers["G_new"], device_buffers["G"]
    )

    device_buffers["B"], device_buffers["B_new"] = (
        device_buffers["B_new"], device_buffers["B"]
    )

@nb.njit(nb.float32[:](nb.float32[:, :]), cache=True)
def _get_backward_seam(energy: np.ndarray) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map"""
    h, w = energy.shape
    inf = np.array([np.inf], dtype=np.float32)
    cost = np.concatenate((inf, energy[0], inf))
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        choices = np.vstack((cost[:-2], cost[1:-1], cost[2:]))
        min_idx = np.argmin(choices, axis=0) + base_idx
        parent[r] = min_idx
        cost[1:-1] = cost[1:-1][min_idx] + energy[r]
    
    c = np.argmin(cost[1:-1])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    print("Seam CPU: \n", seam)

    return cost[1:-1]
