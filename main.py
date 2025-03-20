import numpy as np
import pycuda.driver as cuda
import time
import os
import argparse
import math

from utils.pycuda_utils import get_cuda_kernels_src_file, load_image
from utils.pycuda_utils import get_backward_seam_from_idx

if __name__ == "__main__":

    import pycuda.autoinit

    parser = argparse.ArgumentParser()
    image_name = "joel-filipe-QwoNAhbmLLo-unsplash.jpg"
    default_image_path = os.path.join(os.getcwd(),
                                      "examples", "images", image_name)

    # Arguments provided by user
    parser.add_argument("--img_path", type=str, 
                        help="Path of image to be resized",
                        default=default_image_path)

    parser.add_argument("--resized_width", type=int,
                        help="Width of the resized image",
                        default=400)

    parser.add_argument("--resized_height", type=int,
                        help="Height of the resized image",
                        default=400)

    args = parser.parse_args()

    img_path = args.img_path

    print(default_image_path)

    # Loads img to be resized
    img, image_height, image_width = load_image(img_path)

    # Loads source kernels' file
    ker = get_cuda_kernels_src_file()

    # Gets cuda_kernels 
    rgb_to_gray_with_padding_fn = ker.get_function("Rgb2GrayWithPadding")
    sobel_horizontal_fn = ker.get_function("SobelHorizontal")
    sobel_vertical_fn = ker.get_function("SobelVertical")
    energy_fn_backward = ker.get_function("EnergyMapBackward")
    cumulative_energy_fn_backward = ker.get_function("cumulativeMapBackward")
    min_element_row_fn = ker.get_function("findMinInThreadBlock")
    remove_seam_fn = ker.get_function("removeVerticalSeamAndInsertPadding")

    # Thread Launch Config for rgb2gray and sobel kernels
    threadsPerBlock = (16, 16, 1)
    numBlocks = ((image_width + 2 + threadsPerBlock[0] - 1) // threadsPerBlock[0], 
                 (image_height + 2 + threadsPerBlock[1] - 1) // threadsPerBlock[1], 1)

    # Launch Config for cumulative_energy_fn_backward
    threadsPerBlockCumulative = (1024, 1, 1)
    numBlocksCumulative = (math.ceil(image_width / threadsPerBlockCumulative[0]), 1)

    # Launch Config for seamIdAndShift kernel
    threadsPerBlockSeamId = (1024, 1, 1)
    numBlocksSeamId = (math.ceil(image_width * image_height / threadsPerBlockCumulative[0]), 1)
    print(numBlocksCumulative[0], numBlocksSeamId[0])

    # Launch Config for alterante seamIdAndShift kernel
    threadsPerBlockSeamIdAlternate = (1024, 1, 1)
    numBlocksSeamId = (math.ceil(image_width * image_height / threadsPerBlockCumulative[0]), 1)
    print(numBlocksCumulative[0], numBlocksSeamId[0])

    # Arrays on host 
    gray_image = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)
    gray_image_new = np.zeros((image_height + 2, image_width + 1), dtype=np.uint8)
    sobel_x = np.zeros((image_height, image_width), dtype=np.float32)
    sobel_y = np.zeros((image_height, image_width), dtype=np.float32)
    energy_map = np.zeros((image_height, image_width), dtype=np.float32)
    cumulative_map = np.zeros((image_height, image_width), dtype=np.float32)
    dummy_output = np.zeros((image_height, image_width), dtype=np.int32)
    min_row = np.zeros(image_width, dtype=np.float32)
    seam_indices = np.zeros(image_height, dtype=np.int32)
    min_indices = np.zeros(numBlocksCumulative[0], dtype=np.int32)

    # Allocates arrays on device
    d_sobel_x = cuda.mem_alloc(sobel_x.nbytes)
    d_sobel_y = cuda.mem_alloc(sobel_y.nbytes)
    d_energy_map = cuda.mem_alloc(energy_map.nbytes)
    d_image = cuda.mem_alloc(img.nbytes)
    d_gray = cuda.mem_alloc(gray_image.nbytes)
    d_gray_new = cuda.mem_alloc(gray_image_new.nbytes)
    d_cumulative_energy_map = cuda.mem_alloc(cumulative_map.nbytes)
    d_dummy_output = cuda.mem_alloc(dummy_output.nbytes)
    d_min_row = cuda.mem_alloc(min_row.nbytes)
    d_seam_indices = cuda.mem_alloc(seam_indices.nbytes)
    d_min_indices = cuda.mem_alloc(min_indices.nbytes)

    # Transfers arrays from host to device
    cuda.memcpy_htod(d_sobel_x, sobel_x)
    cuda.memcpy_htod(d_sobel_y, sobel_y)
    cuda.memcpy_htod(d_energy_map, energy_map)
    cuda.memcpy_htod(d_image, img)
    cuda.memcpy_htod(d_gray_new, gray_image_new)
    cuda.memcpy_htod(d_gray, gray_image)
    cuda.memcpy_htod(d_dummy_output, dummy_output)
    cuda.memcpy_htod(d_seam_indices, seam_indices)
    cuda.memcpy_htod(d_min_row, min_row)
    cuda.memcpy_htod(d_min_indices, min_indices)

    start = time.time()

    # Runs RGB to Gray kernel
    start_time = time.time()
    rgb_to_gray_with_padding_fn(d_image, d_gray,
                                np.int32(image_width), np.int32(image_height),
                                block=threadsPerBlock,
                                grid=numBlocks)
    cuda.Context.synchronize()
    end_time = time.time()

    print(f"Rgb2Gray kernel executed in {end_time - start_time}s")

    # Runs Sobel Horizontal kernel
    start_time = time.time()
    sobel_horizontal_fn(d_gray, d_sobel_x,
                        np.int32(image_width), np.int32(image_height),
                        block=threadsPerBlock,
                        grid=numBlocks)
    cuda.Context.synchronize()
    end_time = time.time()

    print("Kernel executed successfully! Time:", end_time - start_time)

    # Runs Sobel vertical kernel
    start_time = time.time()
    sobel_vertical_fn(d_gray, d_sobel_y,
                      np.int32(image_width), np.int32(image_height),
                      block=threadsPerBlock,
                      grid=numBlocks)
    cuda.Context.synchronize()
    end_time = time.time()
    print("Kernel executed successfully! Time:", end_time - start_time)

    # Runs Energy Map Backward kernel
    start_time = time.time()
    energy_fn_backward(d_sobel_x, d_sobel_y, d_energy_map,
                       np.int32(image_width), np.int32(image_height),
                       block=threadsPerBlock,
                       grid=numBlocks)
    cuda.Context.synchronize()
    end_time = time.time()

    print("Kernel executed successfully! Time:", end_time - start_time)

    # Transfer back needed for initializing first row of cumulative energy map
    cuda.memcpy_dtoh(energy_map, d_energy_map)
    cumulative_map[0, :] = energy_map[0, :]

    # Runs cumulativeEnergyBackward kernel
    shared_mem_size = image_height * np.dtype(np.float32).itemsize
    cuda.memcpy_htod(d_cumulative_energy_map, cumulative_map)
    start_time = time.time()
    cumulative_energy_fn_backward(d_energy_map, d_cumulative_energy_map,
                                  d_dummy_output, np.int32(image_height),
                                  np.int32(image_width),
                                  block=threadsPerBlockCumulative,
                                  grid=numBlocksCumulative,
                                  shared=shared_mem_size)
    cuda.Context.synchronize()
    end_time = time.time()
    print("Kernel executed successfully! Time:", end_time - start_time)
    cuda.memcpy_dtoh(cumulative_map, d_cumulative_energy_map)

    min_row = cumulative_map[image_height - 1, :]
    cuda.memcpy_htod(d_min_row, min_row)

    # Runs kernel to find a minimum in every threadblock mapping to a row
    start_time = time.time()
    min_element_row_fn(d_min_row, d_min_indices, np.int32(image_width),
                       block=threadsPerBlockCumulative,
                       grid=numBlocksCumulative)
    cuda.Context.synchronize()
    end_time = time.time()
    print("Kernel executed successfully! Time:", end_time - start_time)

    cuda.memcpy_dtoh(min_indices, d_min_indices)

    # Finds the minimum element in each threadblock, np.argmin reduces them
    min_elements = cumulative_map[image_height - 1, min_indices]
    min_idx = np.argmin(min_elements)
    backtrack_idx = min_indices[min_idx]

    # Finds the minimum energy vertical seam on the host
    start_time = time.time()
    seam = get_backward_seam_from_idx(backtrack_idx,
                                      cumulative_map.reshape(image_height,
                                                             image_width))
    end_time = time.time()
    print(f"Time taken = {end_time - start_time}s")
    print(f"Total time for the pipeline core = {end_time - start}")

    remove_seam_fn(d_seam_indices, d_gray, d_gray_new, np.int32(image_width), np.int32(image_height), 
                   block = threadsPerBlock,
                   grid = numBlocks) 
    
    cuda.Context.synchronize()
    
    gray_removed = np.zeros((image_height + 2, image_width + 1))
    cuda.memcpy_dtoh(gray_removed, d_gray_new)

    print(image_height, image_width, gray_image.shape)
