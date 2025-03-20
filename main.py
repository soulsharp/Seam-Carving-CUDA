from utils import (
    parse_arguments,
    load_image,
    initialize_cuda_kernels,
    allocate_memory,
    run_kernels,
    find_seam,
    remove_seam,
)

def main():
    args = parse_arguments()
    img, image_height, image_width = load_image(args.img_path)

    kernels = initialize_cuda_kernels()
    buffers, device_buffers = allocate_memory(image_height, image_width, img)

    run_kernels(kernels, device_buffers, image_width, image_height)
    find_seam(kernels, buffers, device_buffers, image_width, image_height)
    remove_seam(kernels, device_buffers, image_width, image_height)

if __name__ == "__main__":
    main()