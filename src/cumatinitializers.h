/**
 * @file cumatinitializers.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef CUMATINITIALIZERS_H
#define CUMATINITIALIZERS_H

#include "cumat.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists


/**
 * @brief Get the Empty cuMat object. No Memory allocation is performed.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The cuMat object.
 */
template <typename T>
CuMat<T> getEmptyOnDevice(int width, int height, int channels);


/**
 * @brief Get a cuMat object filled with zeros.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The cuMat object.
 */
template <typename T>
CuMat<T> zerosOnDevice(int width, int height, int channels);


/**
 * @brief Get a cuMat object filled with ones. 
 * Note that you can initialize a cuMat object filled with any value `f` using `onesOnDevice(...) * f`.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The cuMat object.
 */
template <typename T>
CuMat<T> onesOnDevice(int width, int height, int channels);


/**
 * @brief Get a cuMat object filled with ones on the diagonal and zeros elsewhere.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The cuMat object.
 */
template <typename T>
CuMat<T> eyeOnDevice(int width, int height, int channels);

}

#endif  // CUMATINITIALIZERS_H