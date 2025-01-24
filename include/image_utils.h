#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>

unsigned char* load_image(const std::string& filename, int& width, int& height, int& channels);
bool save_image(const std::string& filename, unsigned char* image, int width, int height, int channels);

#endif