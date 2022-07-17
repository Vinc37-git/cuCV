/**
 * @file initializers.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "initializers.h"


template <typename T>
cuCV::Mat<T> cuCV::getEmpty(int width, int height, int channels) {
    return cuCV::Mat<T>(width, height, channels);
}


template <typename T>
cuCV::Mat<T> cuCV::zeros(int width, int height, int channels) {
    cuCV::Mat<T> mat = cuCV::Mat<T>(width, height, channels);
    mat.zeros();
    return mat;
}


template <typename T>
cuCV::Mat<T> cuCV::ones(int width, int height, int channels) {
    cuCV::Mat<T> mat = cuCV::Mat<T>(width, height, channels);
    mat.ones();
    return mat;
}


template <typename T>
cuCV::Mat<T> cuCV::eye(int width, int height, int channels) {
    cuCV::Mat<T> mat = cuCV::Mat<T>(width, height, channels);
    mat.eye();
    return mat;
}


/// Explicit template specialization
template cuCV::Mat<CUCV_8U> cuCV::getEmpty<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_16U> cuCV::getEmpty<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_64F> cuCV::getEmpty<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::Mat<CUCV_8U> cuCV::zeros<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_16U> cuCV::zeros<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_64F> cuCV::zeros<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::Mat<CUCV_8U> cuCV::ones<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_16U> cuCV::ones<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_64F> cuCV::ones<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::Mat<CUCV_8U> cuCV::eye<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_16U> cuCV::eye<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::Mat<CUCV_64F> cuCV::eye<CUCV_64F>(const int width, const int height, const int channels);

