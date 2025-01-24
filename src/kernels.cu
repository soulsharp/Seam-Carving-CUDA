#include <iostream>
#include <cuda_runtime.h>
#include "image_utils.h"
#include "stb_image.h"

// CUDA kernel to double the elements of an array
__global__ void doubleArray(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] *= 2;
    }
}

// This kernel converts a RGB image to Grayscale and adds a zero padding around the Grayscale image
__global__ void Rgb2GrayWithPadding(unsigned char* img, unsigned char* gray_img, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Padding = 1
    int padded_width = width + 2;
    int padded_height = height + 2;

    // Checks if the thread accesses an out of bounds index
    if(x >= padded_width || y >= padded_height){
        return;
    }
    
    // Sets the padding region to 0
    if(x == 0 || x == width + 1 || y == 0 || y == height + 1){
        gray_img[y * padded_width + x] = 0;
        return;
    }

    // Maps to the correct section of the original image
    int idx = ((y-1) * width + (x - 1)) * 3;
    unsigned char red = img[idx];
    unsigned char green = img[idx + 1];
    unsigned char blue = img[idx + 2];

    // calculates grayscale value based on RGB for a pixel
    float gray_val = 0.2989f * red + 0.5870f * green + 0.1140f * blue;

    // Writes the grayscale value back
    gray_img[y * padded_width + x] = static_cast<unsigned char>(gray_val);
}

// This kernel applies the sobel filter on a grayscale image to detect gradients in the X-direction
__global__ void SobelHorizontal(unsigned char* gray_img, float* sobel_x, int width, int height){

    // sobel_x -> height x width
    // gray_img -> (height + 2) x (width + 2)

    // sobel_x - > [[-1, -2, -1], [0, 0, 0], [1, 2, 1]

    // x, y = 0, 0

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // width and height of the gray_img
    int padded_width = width + 2;
    int padded_height = height + 2;  

    if (x <= width - 1 && y <= height - 1) {
        
        int idx = y * width + x;

        // Accessing mid idx of every row involved in the sobel op for better readability
        int gray_top_idx = (y + 0) * padded_width + (x + 1);
        int gray_mid_idx = (y + 1) * padded_width + (x + 1);
        int gray_low_idx = (y + 2) * padded_width + (x + 1);

        // The contribution of the middle row is 0 but is included for readability
        float top_val = - (gray_img[gray_top_idx - 1] + 2 * gray_img[gray_top_idx] + gray_img[gray_top_idx]);
        float mid_val = 0;
        float low_val = gray_img[gray_low_idx - 1] + 2 * gray_img[gray_low_idx] + gray_img[gray_low_idx];

        // Calculates sobel_val for idx 
        float sobel_val = abs(top_val + mid_val + low_val);
        sobel_x[idx] = abs(sobel_val);
    }

}

// This kernel applies the sobel filter on a grayscale image to detect gradients in the Y-direction
__global__ void SobelVertical(unsigned char* gray_img, float* sobel_y, int width, int height) {
    
    // sobel_y -> height x width
    // gray_img -> (height + 2) x (width + 2)

    // sobel_y - > [[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]


    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    // width and height of the gray_img
    int padded_width = width + 2;
    int padded_height = height + 2;  

    if (x <= width - 1 && y <= height - 1) {
        
        int idx = y * width + x;

        // Accessing mid idx of every row involved in the sobel op for better readability
        int gray_top_idx = (y + 0) * padded_width + (x + 1);
        int gray_mid_idx = (y + 1) * padded_width + (x + 1);
        int gray_low_idx = (y + 2) * padded_width + (x + 1);

        // The contribution of the middle row is 0 but is included for readability
        float top_val = -gray_img[gray_top_idx - 1] + 0 + gray_img[gray_top_idx];
        float mid_val = -2 * gray_img[gray_low_idx - 1] + 0 + 2 * gray_img[gray_low_idx];
        float low_val = -gray_img[gray_low_idx - 1] + 0 + gray_img[gray_low_idx];

        // Calculates sobel_val for idx 
        float sobel_val = abs(top_val + mid_val + low_val);
        sobel_y[idx] = abs(sobel_val);
    }

}

__global__ void EnergyMap(float* sobel_x, float* sobel_y, float* energy_map, int width, int height){
    // EnergyMap is just an elementwise sum of sobel_x and sobel_y
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    // Checks if the thread accesses an out of bounds index
    if(x >= width || y >= height){
        return;
    }

    int idx = y * width + x;

    // Calculates EnergyValue corresponding to idx
    float energy_val = sobel_x[idx] + sobel_y[idx];
    energy_map[idx] = energy_val;
}



// Error-checking macro for CUDA calls
#define CUDA_CHECK(call)                                                          \
    do {                                                                           \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"                         \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

int main() {
    // Image specifics
    int width, height, channels;
    std::string img_path = "E:/CUDA Test/images/505616.png";
    std::string save_path = "E:/CUDA Test/images/gray_output.png";

    // std::cout << "Inside main" ;
    
    // Loads image 
    unsigned char* img = load_image(img_path, width, height, channels);
    if (img == nullptr) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return 1;
    }

    size_t gray_img_size = width * height * sizeof(unsigned char);
    unsigned char* gray_ptr;
    CUDA_CHECK(cudaMalloc((void**)&gray_ptr, gray_img_size));
    
    // Saves image
    if (!save_image(save_path, img, width, height, channels)) {
        std::cerr << "Error: Could not save image!" << std::endl;
        return 1;
    }
    
    // Frees space occupied by img
    stbi_image_free(img);
 
    const int size = 10;
    const int bytes = size * sizeof(unsigned char);

    // Host array (CPU)
    int h_array[size];
    for (int i = 0; i < size; i++) {
        h_array[i] = i + 1; // Initialize with values 1, 2, 3, ...
    }

    // Device array (GPU)
    int* d_array;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_array, bytes));

    // Copy the array from host to device
    CUDA_CHECK(cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice));

    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    Rgb2GrayWithPadding<<<numBlocks, threadsPerBlock>>>(img, gray_ptr, width, height);



    // Synchronize to ensure the kernel has finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to the host
    CUDA_CHECK(cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Doubled array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Free the device memory
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
