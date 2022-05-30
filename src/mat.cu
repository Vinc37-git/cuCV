/**
 * @file mat.cu
 * @author your name (vin37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "mat.h"

template <typename T>
cuCV::Mat<T>::Mat() 
        : mData(NULL) { }


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels, T * data) 
        : mWidth(width), mHeight(height), mChannels(channels), mStride(width), mData(data) { }


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels) 
        : mWidth(width), mHeight(height), mChannels(channels), mStride(width), mData(NULL) {
    mData = (T *) malloc(width * height * channels * sizeof(T));
    }


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator+(const Mat & mat) const {
    
    /// Check dimensions
    if (mWidth != mat.mWidth || mHeight != mat.mHeight || mChannels != mat.mChannels) {
        throw "Dimensions do not match!";
    }

    /// @todo Check for datatype equality

    /// @todo Check if mat is empty

    /// Perform Math
    Mat<T> sum(mWidth, mHeight, mChannels);

    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        sum.mData[i] = mData[i] + (T) mat.mData[i];
    }

    return sum;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator+(T alpha) const {
    
    /// @todo Check if mat is empty

    /// Perform Math
    Mat<T> sum(mWidth, mHeight, mChannels);

    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        sum.mData[i] = mData[i] + alpha;
    }

    return sum;
}



template <typename T>
T cuCV::Mat<T>::at(const int row, const int col) const {
    /// row major: a11, a12, a13, a21, ...
    return mData[row * mStride + col];
}


template <typename T>
void cuCV::Mat<T>::ones() {

    if (mData == NULL) { // allocate
        mData = (T *) malloc(mWidth * mHeight * mChannels * sizeof(T));
    }

    for (size_t i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = (T) 1;
    }
}


template <typename T>
void cuCV::Mat<T>::print(int nRows, int nCols) const {
    if (mData == NULL) {
        throw "Data Pointer is NULL.";
    }
    for (int row=0; row < nRows; row++) {
        for (int col=0; col < nCols; col++) {
            printf("%f \t", (double) at(row,col));
        }
        printf("\n");
    }
}



/// Explicit template specialization
template class cuCV::Mat<unsigned char>;
template class cuCV::Mat<unsigned short>;
template class cuCV::Mat<double>;
