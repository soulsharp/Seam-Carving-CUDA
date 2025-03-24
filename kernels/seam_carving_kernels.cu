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
    float grayVal = 0.2125f * red + 0.7154f * green + 0.0721f * blue;

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

    // Writes to sobelX map
    sobelX[idx] = sobelVal;
}

// This kernel applies the sobel filter on a grayscale image to detect gradients in the Y-direction
extern "C" __global__ void SobelVertical(unsigned char* grayImg, float* sobelY, int width, int height) {

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
    int grayMidIdx = paddedY * paddedWidth + paddedX;
    int grayLowIdx = (paddedY + 1) * paddedWidth + paddedX;
    
    // The contribution of the middle column is 0 but is included for readability
    float topVal = -grayImg[grayTopIdx - 1] + 0 + grayImg[grayTopIdx + 1];
    float mid_val = -2 * grayImg[grayMidIdx - 1] + 0 + 2 * grayImg[grayMidIdx + 1];
    float lowVal = -grayImg[grayLowIdx - 1] + 0 + grayImg[grayLowIdx + 1];

    // Calculates sobelVal for idx
    sobelY[idx] = fabsf(topVal + mid_val + lowVal);
}

extern "C" __global__ void EnergyMapBackward(float* sobelX, float* sobelY, float* energyMap, int width, int height){
    // EnergyMap is just an elementwise sum of sobelX and sobel_y
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // Checks if the thread accesses an out of bounds index
    if(x >= width || y >= height) return;

    int idx = y * width + x;

    // Calculates EnergyValue corresponding to idx
    float energyVal = sobelX[idx] + sobelY[idx];
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

extern "C" __global__ void cumulativeMapBackward(float* energyMap, float* cumulativeEnergyMap,
                                int imageHeight, int imageWidth){

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Checks if the thread accesses an out of bounds index
    if(tid >= 1024) return;

    // Starts computation from row no 2
    for(int rowIdx = 1; rowIdx < imageHeight; ++rowIdx){
      for(int pixelIdx = tid; pixelIdx < imageWidth; pixelIdx += 1024){
        int prevRowStartIdx = (rowIdx - 1) * imageWidth;
        int currentRowStartIdx = rowIdx * imageWidth;
        int elementAbove = prevRowStartIdx + pixelIdx;
        int currentElement = currentRowStartIdx + pixelIdx;
        int energyToAdd = 0.0;

        // At the leftmost position
        if(pixelIdx % imageWidth == 0){
            energyToAdd = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove + 1]);
        }

        // At the rightmost position
        else if((pixelIdx + 1) % imageWidth == 0){
            energyToAdd = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove - 1]);
        }

        // Remaining positions
        else{
            int temp = fminf(cumulativeEnergyMap[elementAbove], cumulativeEnergyMap[elementAbove + 1]);
            energyToAdd = fminf(temp, cumulativeEnergyMap[elementAbove - 1]);
        }

        cumulativeEnergyMap[currentElement] = energyMap[currentElement] + energyToAdd;
     }
     __syncthreads();
    }
}

extern "C" __global__ void removeVerticalSeam(int* seamIndices, unsigned char* gray, unsigned char* grayNew,
                                            int energyMapWidth, int energyMapHeight){
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Threads also handle the padding region
    if (tid >= ((energyMapHeight + 2) * (energyMapWidth + 1))) return;
    
    // Row, Col handled by the present thread
    int rowIdx = tid / (energyMapWidth + 1);
    int colIdx = tid % (energyMapWidth + 1);
    
    int seamCol = seamIndices[rowIdx];

    // Amount by which pixels from the old image have to be shifted in the new image
    int amountLeftShift; 
    if (colIdx < seamCol){
        amountLeftShift = rowIdx;
    }
    else{
        amountLeftShift = rowIdx + 1;
    }

    grayNew[tid] = gray[tid + amountLeftShift];
} 


extern "C" __global__ void removeVerticalSeamAndInsertPadding(int* seamIndices, unsigned char* gray, unsigned char* grayNew,
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
        grayNew[grayNewIdx] = 0;
        return;
    }

    // Gets seam index for the current row (adjusted for padding)
    int k = seamIndices[y - 1];

    // Computes the corresponding index in gray
    int grayOldIdx = y * (energyMapWidth + 2) + x;

    // Pixels before the seam pixel remain the same, pixels after get shifted to the left by 1
    if (x < k) {
        grayNew[grayNewIdx] = gray[grayOldIdx];
    }
    else {
        grayNew[grayNewIdx] = gray[grayOldIdx + 1];

    }
}

extern "C" __global__ void removeSeamRGB(unsigned char* red, unsigned char* green, unsigned char* blue,
                                        unsigned char* redNew, unsigned char* greenNew, unsigned char* blueNew, 
                                        int* seamIndices, int width, int height) {

    int y = threadIdx.y + blockIdx.y * blockDim.y; 
    int x = threadIdx.x + blockIdx.x * blockDim.x;  

    // Bounds check
    if (y >= height || x >= width - 1) return;

    int k = seamIndices[y];  
    if (x < k) return;       

    // Shift pixels left
    int idx = y * width + x;
    redNew[idx] = red[idx + 1];
    greenNew[idx] = green[idx + 1];
    blueNew[idx] = blue[idx + 1];
}

extern "C" __global__ void updateEnergyMap(int* seamIndices, unsigned char* grayImg, 
                                         float* sobelX, float* sobelY, float* energyMap, 
                                         int width, int height) {
    // x -> [-1, 0, 1] 
    // y -> [0, height - 1]

    // -1 to shift the range to [-1, 0, 1]
    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check (ensures within valid row indices)
    if (y >= height) return;

    // Computes the column index of the affected pixel
    int seamIdx = seamIndices[y]; 
    int updatedCol = seamIdx + x; 

    // Ensures the updated pixel is within bounds
    if (updatedCol < 1 || updatedCol >= width - 1) return;

    // Converts to padded image coordinates
    int paddedY = y + 1; 
    int paddedWidth = width + 2;
    int paddedIdx = paddedY * paddedWidth + updatedCol;

    // Gets indices for the 3x3 neighborhood needed for Sobel filter
    int topIdx = paddedIdx - paddedWidth;   
    int midIdx = paddedIdx;                 
    int lowIdx = paddedIdx + paddedWidth;  

    // SobelX calculation
    float sobelValX = fabsf(
    -grayImg[topIdx - 1] - 2 * grayImg[midIdx - 1] - grayImg[lowIdx - 1] +
    grayImg[topIdx + 1] + 2 * grayImg[midIdx + 1] + grayImg[lowIdx + 1]
    );

    // SobelY calculation
    float sobelValY = fabsf(
    -grayImg[topIdx - 1] - 2 * grayImg[topIdx] - grayImg[topIdx + 1] +
    grayImg[lowIdx - 1] + 2 * grayImg[lowIdx] + grayImg[lowIdx + 1]
    );

    // Computes energy
    float newEnergy = sobelValX + sobelValY;

    // Computes index in the non-padded outputs
    int outIdx = y * width + updatedCol;
    
    // Modifies maps
    sobelX[outIdx] = sobelValX;
    sobelY[outIdx] = sobelValY;
    energyMap[outIdx] = newEnergy;
}