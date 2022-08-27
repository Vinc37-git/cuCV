/**
 * @file initializers.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <cmath>
#include "mat.h"
#include "errorhandling.h"

namespace cuCV {


/**
 * @brief Get the Empty Mat object.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The Mat object.
 */
template <typename T>
Mat<T> getEmpty(int width, int height, int channels);


/**
 * @brief Get a Mat Object filled with zeros.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The Mat object.
 */
template <typename T>
Mat<T> zeros(int width, int height, int channels);


/**
 * @brief Get a Mat object filled with ones.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The Mat object.
 */
template <typename T>
Mat<T> ones(int width, int height, int channels);


/**
 * @brief Get a mat object filled with ones on the diagonal and zeros elsewhere.
 * 
 * @tparam T The type of the matrix.
 * @param width The width of the matrix.
 * @param height The height of the matrix
 * @param channels The number of channels of the matrix.
 * @return The Mat object.
 */
template <typename T>
Mat<T> eye(int width, int height, int channels);


/**
 * @brief Get a squared mat object which entries follow a gaussian distribution.
 * 
 * @tparam T The type of the matrix.
 * @param length The length of one side of the matrix.
 * @param channels The number of channels of the matrix.
 * @param sigma The standard deviation. Defaults to 1.
 * @param norm Normalize kernel such that the sum over all elements equals 1. Defaults to true.
 * @return The Mat object.
 */
template <typename T>
Mat<T> gauss(const int length, const int channels, double sigma=1, bool norm=true);

}

#endif  // INITIALIZERS_H