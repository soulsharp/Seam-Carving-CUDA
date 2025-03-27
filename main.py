import pycuda.driver as cuda
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils.utils import (
    parse_arguments,
    load_image,
    initialize_cuda_kernels,
    allocate_memory,
    run_kernels,
    find_seam,
    remove_seam,
    update_energy_map,
    remove_seam_from_RGB,
    save_image,
)

def process_seams(kernels, buffers, device_buffers, image_width, image_height, target_width, seam_buffer, pbar):
    """Removes seams iteratively until the image reaches the target width."""
    counter = 0

    while image_width > target_width:
        # Calculates cumulative energy map and finds the minimum energy seam
        find_seam(kernels, buffers, seam_buffer, device_buffers, image_width, image_height)

        # Removes seam from the gray image and energy, sobel maps
        remove_seam(kernels, device_buffers, image_width, image_height)

        # Removes seam from R, G, B channels
        remove_seam_from_RGB(kernels, device_buffers, image_width, image_height)
        image_width -= 1

        # Updates the energy and sobel maps 
        update_energy_map(kernels, device_buffers, image_width, image_height)

        pbar.update(1)

    return image_width

def extract_and_transpose_channels(device_buffers, image_height, image_width, flag_transpose=True, flag_rgb=True):
    """Extracts R, G, B channels from device buffers and transposes them."""
    channels = {color: np.zeros((image_height, image_width), dtype=np.uint8) for color in ["R", "G", "B"]}
    for color in channels:
        cuda.memcpy_dtoh(channels[color], device_buffers[color])
        
        if flag_transpose:
            channels[color] = np.transpose(channels[color], (1, 0))
    
    if flag_rgb:
        return np.ascontiguousarray(np.stack((channels["R"], channels["G"], channels["B"]), axis=-1))
    
    else:
        return np.ascontiguousarray(np.stack((channels["B"], channels["G"], channels["R"]), axis=-1))


def main():
    args = parse_arguments()
    img, image_height, image_width = load_image(args.img_path)
    kernels = initialize_cuda_kernels()
    buffers, device_buffers = allocate_memory(image_height, image_width, img)
    seam_vertical = np.zeros(image_height + 2, dtype=np.int32)
    seam_horizontal = np.zeros(args.resized_width + 2, dtype=np.int32)
    
    # Tracks progress of the algorithm
    total_seams_width = max(image_width - args.resized_width, 0)
    total_seams_height = max(image_height - args.resized_height, 0)
    
    with tqdm(total=total_seams_width, desc="Vertical Seam Removal Progress", unit="seam") as pbar_vertical:
        if args.resized_width < image_width:
            run_kernels(kernels, device_buffers, image_width, image_height)
            image_width = process_seams(kernels, buffers, device_buffers, image_width, image_height, args.resized_width, seam_vertical, pbar_vertical)
            print("Removed all vertical seams...")
        elif args.resized_width > image_width:
            print("Image upsizing is not supported at the moment. Exiting...")
            return
        
    with tqdm(total=total_seams_height, desc="Horizontal Seam Removal Progress", unit="seam") as pbar_horizontal:
        if args.resized_height < image_height:
            img_transposed = extract_and_transpose_channels(device_buffers, image_height, image_width)
            image_height, image_width = image_width, image_height
            print(image_height, image_width)
            buffers, device_buffers = allocate_memory(image_height, image_width, img_transposed)
            run_kernels(kernels, device_buffers, image_width, image_height)
            image_width = process_seams(kernels, buffers, device_buffers, image_width, image_height, args.resized_height, seam_horizontal, pbar_horizontal)
            print("Removed all horizontal seams...")
        elif args.resized_height > image_height:
            print("Image upsizing is not supported at the moment. Exiting...")
            return
        
    if(args.resized_height == image_height):
        flag_transpose = False
    else:
        flag_transpose = True

    img_out = extract_and_transpose_channels(device_buffers, image_height, image_width, flag_transpose=flag_transpose, flag_rgb=False)
    save_image(img_out, args.img_path)
    print("Resized image shape:", img_out.shape)

if __name__ == "__main__":
    main()