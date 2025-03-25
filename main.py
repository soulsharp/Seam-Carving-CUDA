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
    update_energy_map,
    remove_seam_from_RGB,
    _get_backward_seam,
)

def main():
    args = parse_arguments()
    img, image_height, image_width = load_image(args.img_path)
    print(type(img))

    kernels = initialize_cuda_kernels()
    buffers, device_buffers = allocate_memory(image_height, image_width, img)
    seam = np.zeros(image_height + 2, dtype=np.int32)
    energy_output = np.zeros(image_height * (image_width - 1), dtype=np.float32)
    
    # If resized_width < image_width, remove vertical seams, else, add them
    if args.resized_width < image_width:
        run_kernels(kernels, device_buffers, image_width, image_height)
        counter = 0

        # Shifts
        shifted_gray_image = np.zeros((image_height + 2, image_width + 2), dtype=np.uint8)
        cuda.memcpy_dtoh(shifted_gray_image, device_buffers["gray_image_new"])
        seam_element = np.array([2000], dtype=np.int32)
        seam_dummy = np.broadcast_to(seam_element, (image_height + 2,))
        cuda.memcpy_htod(device_buffers["seam_indices"], np.ascontiguousarray(seam_dummy))

        cuda.memcpy_dtoh(shifted_gray_image, device_buffers["gray_image"])

        flag = False
        while(image_width > args.resized_width):
                # find_seam(kernels, buffers, seam, device_buffers, image_width, image_height)
                # dummy_energy = np.zeros((image_height, image_width), dtype=np.float32)
                # cuda.memcpy_dtoh(dummy_energy, device_buffers["energy_map"])
                # cost = _get_backward_seam(dummy_energy)
                inter_gray = np.zeros((image_height + 2, image_width + 1), dtype=np.uint8)
                inter_red = np.zeros((image_height, image_width), dtype=np.uint8)
                inter_green = np.zeros((image_height, image_width), dtype=np.uint8)
                inter_blue = np.zeros((image_height, image_width), dtype=np.uint8) 

                remove_seam(kernels, device_buffers, image_width, image_height)

                cuda.memcpy_dtoh(inter_gray, device_buffers["gray_image"])
                cuda.memcpy_dtoh(inter_red, device_buffers["R"])
                cuda.memcpy_dtoh(inter_green, device_buffers["G"])
                cuda.memcpy_dtoh(inter_blue, device_buffers["B"])

                print("Inter red: \n", inter_red)
                print("Inter blue: \n", inter_blue)
                remove_seam_from_RGB(kernels, device_buffers, image_width, image_height)
                image_width -= 1
 
                
                # Update energy map function takes the image_width reduced by 1 
                # update_energy_map(kernels, device_buffers, image_width, image_height, flag)

                flag = not flag
                counter += 1
                if(counter % 5 == 0):
                    print(f"Iteration {counter}")
                
                # print(shifted_gray_image)
                
                
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
    gray_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)


    # cuda.memcpy_dtoh(shifted_gray_image, device_buffers["gray_image"])
    # print(gray_out)
    # cuda.memcpy_dtoh(R_out, device_buffers["R_new"])
    # cuda.memcpy_dtoh(G_out, device_buffers["G_new"])
    # cuda.memcpy_dtoh(B_out, device_buffers["B_new"])

    # print(np.unique(img[:, :, 0] == R_out, return_counts=True))
    # print(np.unique(img[:, :, 0] == G_out, return_counts=True))
    # print(np.unique(img[:, :, 0] == B_out, return_counts=True))

    img_out = np.stack((inter_blue, inter_green, inter_red), axis=-1)

    # print(img_out[: args.resized_height, : args.resized_width].shape)

    cv.imwrite("output.jpg", img_out)
    print(shifted_gray_image)
    cv.imwrite("grayscale_reduced.jpg", inter_gray)

    # # print(img_out[1, :])

if __name__ == "__main__":
    main()