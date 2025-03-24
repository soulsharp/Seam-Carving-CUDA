import pycuda.driver as cuda
import numpy as np
import cv2 as cv

from utils.utils import (
    parse_arguments,
    load_image,
    initialize_cuda_kernels,
    allocate_memory,
    run_kernels,
    find_seam,
    remove_seam,
    remove_seam_alternate,
    update_energy_map,
    remove_seam_from_RGB,
    _get_backward_seam,
)

def main():
    args = parse_arguments()
    img, image_height, image_width = load_image(args.img_path)
    print(img.shape)

    kernels = initialize_cuda_kernels()
    buffers, device_buffers = allocate_memory(image_height, image_width, img)
    
    # If resized_width < image_width, remove vertical seams, else, add them
    if args.resized_width < image_width:
        run_kernels(kernels, device_buffers, image_width, image_height)
        dummy_output = np.zeros_like(buffers["energy_map"], dtype = np.float32)
        dummy_cumulative_last_row = np.zeros(image_width, dtype=np.float32)
        cuda.memcpy_dtoh(dummy_output, device_buffers["energy_map"])
        dummy_cumulative_last_row = _get_backward_seam(dummy_output)
        print("Last_row_of_CPU_implementation cumulative map", dummy_cumulative_last_row)
        # cumulative_energy_CPU =  cumulative_map_backward(dummy_output)
        # print("Last row of cumulative energy map CPU", cumulative_energy_CPU[-1, :])
        # print("Seam from CPU implementation: ", seam)
        counter = 0
        flag = False
        while(image_width > args.resized_width):
                last_row = find_seam(kernels, buffers, device_buffers, image_width, image_height)
                print(np.unique((last_row == dummy_cumulative_last_row), return_counts=True))
                

                # before_seam_removal = np.zeros_like(buffers["gray_image"], dtype=np.uint8)
                # after_seam_removal = np.zeros_like(buffers["gray_image"], dtype=np.uint8)
                # cuda.memcpy_dtoh(before_seam_removal, device_buffers["gray_image"])
                # print(f"Before seam removal: \n{before_seam_removal}")
                # remove_seam_alternate(kernels, device_buffers, image_width, image_height, not flag)
                # cuda.memcpy_dtoh(after_seam_removal, device_buffers["gray_image"])
                # # print(after_seam_removal.ravel()[0 :((image_height + 2) * (image_width + 1))].reshape(image_height + 2, image_width + 1))
                # print(f"After seam removal: \n{after_seam_removal}")

                # print(np.unique(before_seam_removal==after_seam_removal, return_counts=True))
                
                # remove_seam_from_RGB(kernels, device_buffers, image_width, image_height)
                # update_energy_map(kernels, device_buffers, image_width, image_height, flag)
                image_width -= 1
                # flag = not flag
                # counter += 1
                # if(counter % 5 == 0):
                #     print(f"Iteration {counter}")
                
                
    # elif args.resized_width > image_width:
    #     print("Image upsizing is not supported at the moment.Exiting...")
    #     return

    # # If resized_height > image_height, remove vertical seams from transpose of the image, else, add them
    # if args.resized_height < image_height:
    #     pass
    # elif args.resized_height > image_height:
    #     print("Image upsizing is not supported at the moment.Exiting...")
    #     return
    
    # # run_kernels(kernels, device_buffers, image_width, image_height)
    # # a = np.zeros_like(buffers["energy_map"])
    # # cuda.memcpy_dtoh(a, device_buffers["energy_map"])
    # # find_seam(kernels, buffers, device_buffers, image_width, image_height)
    # # remove_seam(kernels, device_buffers, image_width, image_height)
    # # update_energy_map(kernels, device_buffers, image_width, image_height)

    # R_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # G_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # B_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # gray_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # cuda.memcpy_dtoh(gray_out, device_buffers["gray_image"])
    # cuda.memcpy_dtoh(R_out, device_buffers["R_new"])
    # cuda.memcpy_dtoh(G_out, device_buffers["G_new"])
    # cuda.memcpy_dtoh(B_out, device_buffers["B_new"])

    # print(np.unique(img[:, :, 0] == R_out, return_counts=True))
    # print(np.unique(img[:, :, 0] == G_out, return_counts=True))
    # print(np.unique(img[:, :, 0] == B_out, return_counts=True))

    # img_out = np.stack((B_out, G_out, R_out), axis=-1)

    # print(img_out[: args.resized_height, : args.resized_width].shape)

    # cv.imwrite("output.jpg", img_out[: args.resized_height, : args.resized_width])
    # cv.imwrite("gray.jpg", gray_out[: args.resized_height, : args.resized_width])

    # # print(img_out[1, :])

if __name__ == "__main__":
    main()