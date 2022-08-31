/**
 * @file imageloader.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "imread.h"


cuCV::Mat<CUCV_8U> cuCV::imread(const char * path) {

    if (!std::filesystem::is_regular_file(path)) {
        printf("No file found at '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    
    cimg_library::CImg<unsigned char> image;

    image.load(path);

    size_t counts = image.width() * image.height() * image.depth() * sizeof(unsigned char);

    unsigned char * data = new unsigned char [counts];
    memcpy((void *) data, (const void *) image.data(), counts);
    CUCV_DEBUG_PRINT("Copied %p to %p : imread.\n", image.data(), data);
    
    // The mat will take care about free(data) / delete [] data
    // The compiler will handle mat as lvalue and hence not copy the data a second time.
    return cuCV::Mat<CUCV_8U>(image.width(), image.height(), image.depth(), data);
}


cuCV::CuMat<CUCV_8U> cuCV::imreadToDevice(const char * path) {
    if (!std::filesystem::is_regular_file(path)) {
        printf("No file found at '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    
    cimg_library::CImg<unsigned char> image;
    image.load(path);
    size_t counts = image.width() * image.height() * image.depth() * sizeof(unsigned char);

    /// Allocate Memory
    cuCV::CuMat<CUCV_8U> cuMat(image.width(), image.height(), image.depth());
    cuMat.allocateOnDevice();

    gpuErrchk(cudaMemcpy((void *) cuMat.getDataPtr(), (void *) image.data(), counts, cudaMemcpyHostToDevice));   

    return cuMat;
}


void cuCV::imwrite(cuCV::Mat<CUCV_8U> & mat, const char * path) {
    
    cimg_library::CImg<unsigned char> image(mat.getDataPtr(), mat.getWidth(), mat.getHeight(), mat.getNChannels(), true);

    image.save(path);
}


void cuCV::imwrite(cuCV::CuMat<CUCV_8U> & cuMat, const char * path) {

    cuCV::Mat<CUCV_8U> mat;
    cuMat.downloadTo(mat);
    
    cuCV::imwrite(mat, path);
}

