#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



#include "image_utils.h" 
#include <iostream>

// Uses stbi_load to load an image specified as load_path
unsigned char* load_image(const std::string& load_path, int& width, int& height, int& channels) {
    return stbi_load(load_path.c_str(), &width, &height, &channels, 0);
}

// Uses stbi_write_png to save an image specified as save_path
bool save_image(const std::string& save_path, unsigned char* image, int width, int height, int channels) {
    return stbi_write_png(save_path.c_str(), width, height, channels, image, width * channels);

std::cout << "Inside load and save images";


// void stbi_free(void *retval) {
//     stbi_image_free(retval);
// }
}
