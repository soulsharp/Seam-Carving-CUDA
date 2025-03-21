import pycuda.driver as cuda

from utils.utils import (
    parse_arguments,
    load_image,
    initialize_cuda_kernels,
    allocate_memory,
    run_kernels,
    find_seam,
    remove_seam,
    update_energy_map
)

def main():
    args = parse_arguments()
    img, image_height, image_width = load_image(args.img_path)

    kernels = initialize_cuda_kernels()
    buffers, device_buffers = allocate_memory(image_height, image_width, img)
    
    # # If resized_width < image_width, remove vertical seams, else, add them
    # if args.resized_width < image_width:
    #     while(image_width >= args.resized_width):
    #             run_kernels(kernels, device_buffers, image_width, image_height)
    #             find_seam(kernels, buffers, device_buffers, image_width, image_height)
    #             remove_seam(kernels, device_buffers, image_width, image_height)
    #             image_width -= 1
                
    # elif args.resized_width > image_width:
    #     print("Image upsizing is not supported at the moment.Exiting...")
    #     return

    # # If resized_height > image_height, remove vertical seams from transpose of the image, else, add them
    # if args.resized_height < image_height:
    #     pass
    # elif args.resized_height > image_height:
    #     print("Image upsizing is not supported at the moment.Exiting...")
    #     return
    
    run_kernels(kernels, device_buffers, image_width, image_height)
    find_seam(kernels, buffers, device_buffers, image_width, image_height)
    remove_seam(kernels, device_buffers, image_width, image_height)
    update_energy_map(kernels, device_buffers, image_width, image_height)

if __name__ == "__main__":
    main()