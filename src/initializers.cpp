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


double gaussian(double x, double mu, double sigma) {
    const double a = (x - mu) / sigma;
    return std::exp(-0.5 * a * a);    
}


template <typename T>
cuCV::Mat<T> cuCV::gauss(const int length, const int channels, double sigma, bool norm) {
    /// @todo Throw proper exception. 
    if (length % 2 == 0) 
        throw std::runtime_error("Invalid side-length for Gaussian Kernel. It must be an odd number."); 
    
    //T * data = (T *) malloc(length * length * channels * sizeof(T));
    T * data = new T [length * length * channels];

    const int radius = length / 2;

    //sigma = radius/2.f;

    for (size_t ch = 0; ch < channels; ch++) {
        double sum = 0;
        for (size_t row = 0; row < length; row++) {
            for (size_t col = 0; col < length; col++) {
                double out = gaussian(row, radius, sigma) * gaussian(col, radius, sigma);
                data[row * length + col + (length * length) * ch] = (T) out;
                sum += out;
            }
        }
        if (norm)
            // normalize channel-wise
            for (size_t i = 0; i < length * length; ++i) {
                double out = data[i + (length * length) * ch] / sum;
                data[i + (length * length) * ch] = (T) out;
            }
    }
    return cuCV::Mat<T>(length, length, channels, data);
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

//template cuCV::Mat<CUCV_8U> cuCV::gauss<CUCV_8U>(const int length, const int channels, const int sigma);
// template cuCV::Mat<CUCV_16U> cuCV::gauss<CUCV_16U>(const int length, const int channels, const int sigma);
// template cuCV::Mat<CUCV_64F> cuCV::gauss<CUCV_64F>(const int length, const int channels, const int sigma);
// template cuCV::Mat<CUCV_8U> cuCV::gauss(int length, int channels, double sigma, bool norm);
// template cuCV::Mat<CUCV_16U> cuCV::gauss(int length, int channels, double sigma, bool norm);
template cuCV::Mat<CUCV_64F> cuCV::gauss(int length, int channels, double sigma, bool norm);
