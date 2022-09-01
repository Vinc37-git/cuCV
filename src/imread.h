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
#define cimg_use_png 1
#define cimg_display 0  // then we do not need to link X11 against executables
#include "include/cimg/CImg.h"
#include "mat.h"
#include "cumat.h"
#include "errorhandling.h"

namespace cuCV {


/**
 * @brief Reads an image at `path` and returns it as a cuCV::Mat of type CUCV_8U.
 * CImg is used as an API to load the images. Afterwards, a mat object is created
 * from the data array of the CImg image instance. Note that this function will copy
 * the whole data of the loaded image once, since the CImg object will delete all loaded
 * data after this function call.
 * 
 * Allowed image are all that CImg supports. However, note that besides loading PNG and JPEG,
 * you must link your applications against the respictive library. See the CImg documentation
 * for further information. 
 * 
 * @param path the path of the image 
 * @return Mat<CUCV_8U> 
 */
Mat<CUCV_8U> imread(const char * path);


/**
 * @brief Reads an image at `path`, uploads it to the device and returns in as a cuCV::CuMat of 
 * type CUCV_8U. Besides that the data is uploaded directly to the device and not copied on the 
 * host, it works like `imread`.
 * 
 * @param path 
 * @return CuMat<CUCV_8U> 
 */
CuMat<CUCV_8U> imreadToDevice(const char * path);


/**
 * @brief Write an instance of Mat to the disk. The write format is guessed from the path ending.
 * CImg is used as an API to write images. CImg objects are created from the mat buffer
 * and saved to the disk. As format, you can use any format that is available from CImg. However,
 * note that besides from PNG and JPEG you must link the respective library against your application.
 * 
 * @param mat The mat object that should be written.
 * @param path The path, where the images should be saved.
 */
void imwrite(Mat<CUCV_8U> & mat, const char * path);


/**
 * @brief Write an instance of CuMat to the disk. The write format is guessed from the path ending.
 * CImg is used as an API to write images. The CuMat buffer is automatically downloaded to the host
 * and then, CImg objects are created from the buffer
 * and saved to the disk. As format, you can use any format that is available from CImg. However,
 * note that besides from PNG and JPEG you must link the respective library against your application.
 * 
 * @param mat The mat object that should be written.
 * @param path The path, where the images should be saved.
 */
void imwrite(CuMat<CUCV_8U> & cuMat, const char * path);

}

#endif  // IMREAD_H


