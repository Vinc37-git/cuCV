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
        fprintf(stderr, "Error: No file found at '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    
    // Load image with CImg. Data will be stored in image and CImg does not allow us to steal the data.
    cimg_library::CImg<unsigned char> image;
    image.load(path);
    size_t counts = image.width() * image.height() * image.depth() * sizeof(unsigned char);

    CUCV_DEBUG_PRINT("Read image to %p.", image.data());

    // Create a borrowed reference to image
    cuCV::Mat<CUCV_8U> mat(image.width(), image.height(), image.depth(), image.data(), true);
    
    // ... and return a copy of mat what will copy the image data and own it afterwards
    //return cuCV::Mat<CUCV_8U> (cuCV::Mat<CUCV_8U> (image.width(), image.height(), image.depth(), image.data(), true));
    return cuCV::Mat<CUCV_8U> (mat);
}


cuCV::CuMat<CUCV_8U> cuCV::imreadToDevice(const char * path) {
    if (!std::filesystem::is_regular_file(path)) {
        fprintf(stderr, "Error: No file found at '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    
    cimg_library::CImg<unsigned char> image;
    image.load(path);
    size_t counts = image.width() * image.height() * image.depth() * sizeof(unsigned char);

    CUCV_DEBUG_PRINT("Read image to %p.", image.data());

    /// Allocate Memory
    cuCV::CuMat<CUCV_8U> cuMat(image.width(), image.height(), image.depth());
    cuMat.allocateOnDevice();

    /// upload memory. @todo: write method `uploadFrom(T * data, size_t counts);
    gpuErrchk(cudaMemcpy((void *) cuMat.getDataPtr(), (void *) image.data(), counts, cudaMemcpyHostToDevice));   

    CUCV_DEBUG_PRINT("Uploaded %ld bytes to %p. ", counts, cuMat.getDataPtr());

    return cuMat;
}


void cuCV::imwrite(cuCV::Mat<CUCV_8U> & mat, const char * path) {
    
    cimg_library::CImg<unsigned char> image(mat.getDataPtr(), mat.getWidth(), mat.getHeight(), mat.getNChannels(), true);

    image.save(path);

    CUCV_DEBUG_PRINT("Wrote image to %s", path);
}


void cuCV::imwrite(cuCV::CuMat<CUCV_8U> & cuMat, const char * path) {

    cuCV::Mat<CUCV_8U> mat;
    cuMat.downloadTo(mat);
    
    cuCV::imwrite(mat, path);
}

