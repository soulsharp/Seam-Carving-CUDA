#include <iostream>
#include "stb_image_write.h"
#include "image_utils.h" 
#include "stb_image.h"
#include "stb_image_write.h"


int main() {

    // Image specifics
    int width, height, channels;
    std::string img_path = "E:/CUDA Test/images/505616.png";
    std::string save_path = "E:/CUDA Test/images/saved505616.png";

    std::cout << "Inside main" ;

    unsigned char* img = load_image(img_path, width, height, channels);
    if (img == nullptr) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return 1;
    }

    if (!save_image("output.png", img, width, height, channels)) {
        std::cerr << "Error: Could not save image!" << std::endl;
        return 1;
    }

    stbi_image_free(img);
 
    return 0;
}