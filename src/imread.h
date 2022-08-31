/**
 * @file imread.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef IMREAD_H
#define IMREAD_H

#include <filesystem>

#define cimg_use_jpeg 1
//#define cimg_use_png 1
#define cimg_display 0  // then we do not need to link X11 against executables
#include "include/cimg/CImg.h"



//#include "ImageMagick-6/magick/magick-baseconfig.h"
// #include "ImageMagick-6/magick/magick-config.h"
// #include "ImageMagick-6/Magick++/Include.h"
// #include "ImageMagick-6/Magick++/Image.h"

// #include "Magick++.h"

#include "mat.h"
#include "cumat.h"
#include "errorhandling.h"

namespace cuCV {


/**
 * @brief Reads an image at `path` and returns it as a cuCV::Mat of type T.
 * 
 * @tparam T 
 * @param path 
 * @return Mat<T> 
 */
Mat<CUCV_8U> imread(const char * path);


/**
 * @brief Reads an image at `path`, uploads it to the device and returns in as a cuCV::CuMat of type T.
 * 
 * @tparam T 
 * @param path 
 * @return CuMat<T> 
 */
CuMat<CUCV_8U> imreadToDevice(const char * path);


//template <typename T>
void imwrite(Mat<CUCV_8U> & mat, const char * path);

void imwrite(CuMat<CUCV_8U> & cuMat, const char * path);

}

#endif  // IMREAD_H


