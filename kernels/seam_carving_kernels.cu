# include <cmath>
# include <cfloat>

extern "C" __global__ void Rgb2GrayWithPadding(unsigned char* img, unsigned char* grayImg, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Padding = 1
    int paddedWidth = width + 2;
    int paddedHeight = height + 2;

    // Checks if the thread accesses an out of bounds index
    if(x >= paddedWidth || y >= paddedHeight){
        return;
    }

    // Sets the padding region to 0
    if(x == 0 || x == width + 1 || y == 0 || y == height + 1){
        grayImg[y * paddedWidth + x] = 0;
        return;
    }

    // Maps to a section of the original image
    int idx = ((y-1) * width + (x - 1)) * 3;
    unsigned char red = img[idx];
    unsigned char green = img[idx + 1];
    unsigned char blue = img[idx + 2];

    // calculates grayscale value based on RGB values for a pixel
    float grayVal = 0.2989f * red + 0.5870f * green + 0.1140f * blue;

    // Writes the grayscale value back
    grayImg[y * paddedWidth + x] = static_cast<unsigned char>(grayVal);
}

// This kernel applies the sobel filter on a grayscale image to detect gradients in the X-direction
extern "C" __global__ void SobelHorizontal(unsigned char* grayImg, float* sobelX, int width, int height) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Accessing corresponding position in padded image
    int paddedX = x + 1;
    int paddedY = y + 1;
    int paddedWidth = width + 2;

    // Accessing mid idx of every row involved in the Sobel operation
    int grayTopIdx = (paddedY - 1) * paddedWidth + paddedX;
    int grayLowIdx = (paddedY + 1) * paddedWidth + paddedX;

    // Sobel calculation
    float topVal = -grayImg[grayTopIdx - 1] - 2 * grayImg[grayTopIdx] - grayImg[grayTopIdx + 1];
    float lowVal =  grayImg[grayLowIdx - 1] + 2 * grayImg[grayLowIdx] + grayImg[grayLowIdx + 1];

    float sobelVal = fabsf(topVal + lowVal);
    sobelX[idx] = sobelVal;
}

// This kernel applies the sobel filter on a grayscale image to detect gradients in the Y-direction
extern "C" __global__ void SobelVertical(unsigned char* grayImg, float* sobel_y, int width, int height) {

    // sobel_y -> height x width
    // grayImg -> (height + 2) x (width + 2)

    // sobel_y - > [[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]]

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height) return;

    // Accessing corresponding position in padded image
    int paddedX = x + 1;
    int paddedY = y + 1;

    // Width of the grayImg
    int paddedWidth = width + 2;

    // Index into Sobel_Y
    int idx = y * width + x;

    // Accessing mid idx of every row involved in the Sobel operation
    int grayTopIdx = (paddedY - 1) * paddedWidth + paddedX;
    int grayLowIdx = (paddedY + 1) * paddedWidth + paddedX;

    // The contribution of the middle row is 0 but is included for readability
    float topVal = -grayImg[grayTopIdx - 1] + 0 + grayImg[grayTopIdx];
    float mid_val = -2 * grayImg[grayLowIdx - 1] + 0 + 2 * grayImg[grayLowIdx];
    float lowVal = -grayImg[grayLowIdx - 1] + 0 + grayImg[grayLowIdx];

    // Calculates sobelVal for idx
    float sobelVal = fabsf(topVal + mid_val + lowVal);
    sobel_y[idx] = abs(sobelVal);
}

extern "C" __global__ void EnergyMapBackward(float* sobelX, float* sobel_y, float* energyMap, int width, int height){
    // EnergyMap is just an elementwise sum of sobelX and sobel_y
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // Checks if the thread accesses an out of bounds index
    if(x >= width || y >= height) return;

    int idx = y * width + x;

    // Calculates EnergyValue corresponding to idx
    float energyVal = sobelX[idx] + sobel_y[idx];
    energyMap[idx] = energyVal;
}

extern "C" __global__ void findMinInThreadBlock(float* inputRow, int* minIndices, int length) {
    __shared__ float minValuesShared[1024];
    __shared__ int minIndicesShared[1024];

    int tid = threadIdx.x;
    int globalTid = tid + blockDim.x * blockIdx.x;

    // Loads elements into shared memory
    if (globalTid < length) {
        minValuesShared[tid] = inputRow[globalTid];
        minIndicesShared[tid] = globalTid;
    } else {
        minValuesShared[tid] = FLT_MAX;
        minIndicesShared[tid] = -1;
    }
    __syncthreads();

    // Parallel reduction to find min value and corresponding index
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int right = tid + s;
            if (right < blockDim.x && minValuesShared[right] < minValuesShared[tid]) {
                minValuesShared[tid] = minValuesShared[right];
                minIndicesShared[tid] = minIndicesShared[right];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        minIndices[blockIdx.x] = minIndicesShared[0];
    }
}

extern "C" __global__ void cumulativeMapBackward(float* energyMap, float* cumulativeEnergyMap, int* dummyOutput,
                                int imageHeight, int imageWidth){

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Checks if the thread accesses an out of bounds index
    if(tid >= imageWidth) return;

    // Starts computation from row no 2
    for(int rowIdx = 1; rowIdx < imageHeight; ++rowIdx){
      int prevRowStartIdx = (rowIdx - 1) * imageWidth;
      int currentRowStartIdx = rowIdx * imageWidth;
      int elementAbove = prevRowStartIdx + tid;
      int currentElement = currentRowStartIdx + tid;
      int energyToAdd = 0.0;

      // At the leftmost position
      if(tid % imageWidth == 0){
        energyToAdd = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove + 1]);
      }

      // At the rightmost position
      else if((tid + 1) % imageWidth == 0){
        energyToAdd = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove - 1]);
      }

      // Remaining positions
      else{
        int temp = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove + 1]);
        energyToAdd = fminf(temp, cumulativeEnergyMap[elementAbove - 1]);
      }

      cumulativeEnergyMap[currentElement] = energyMap[currentElement] + energyToAdd;

      __syncthreads();
    }
}

extern "C" __global__ void removeVerticalSeamAndInsertPadding(int* seamIndices, float* gray, float* grayNew,
                                                            int energyMapWidth, int energyMapHeight) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check
    if (x >= energyMapWidth + 1 || y >= energyMapHeight + 2) {
        return;
    }

    // Index in grayNew where this thread writes
    int grayNewIdx = y * (energyMapWidth + 1) + x;

    // Sets padding outside the actual content region to 0
    if (x == 0 || x == energyMapWidth || y == 0 || y == energyMapHeight + 1) {
        grayNew[grayNewIdx] = 0.0;
        return;
    }

    // Gets seam index for the current row (adjusted for padding)
    int k = seamIndices[y - 1];

    // Computes the corresponding index in gray
    int grayOldIdx = y * (energyMapWidth + 2) + x;

    // Pixels before the seam pixel remain the same, pixels after get shifted to the left by 1
    if (x <= k) {
        grayNew[grayNewIdx] = gray[grayOldIdx];
    }
    else {
        grayNew[grayNewIdx] = gray[grayOldIdx + 1];
    }
}
