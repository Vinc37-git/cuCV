/**
 * @file mat.cpp
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
        : mWidth(0), mHeight(0), mChannels(0), mStride(0), mData(NULL) { }


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels, T * data) 
        : mWidth(width), mHeight(height), mChannels(channels), mStride(width), mData(data) { }


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels) 
        : mWidth(width), mHeight(height), mChannels(channels), mStride(width), mData(NULL) {
    //mData = (T *) malloc(width * height * channels * sizeof(T));
    }


template <typename T>
cuCV::Mat<T>::Mat(const Mat & mat) 
        : mWidth(mat.mWidth), mHeight(mat.mHeight), mChannels(mat.mChannels), mStride(mat.mStride), mData(NULL) {
    if (mat.mData != NULL) {
        mData = (T *) malloc(mWidth * mHeight * mChannels * sizeof(T));
        memcpy(mData, mat.mData, mWidth * mHeight * mChannels * sizeof(T));
    }
    //fprintf(stdout, "Copy constructor used.\n");
}


template <typename T>
cuCV::Mat<T>::Mat(Mat && mat) 
        : mWidth(mat.mWidth), mHeight(mat.mHeight), mChannels(mat.mChannels), mStride(mat.mStride), mData(mat.mData) {
    mat.mData = NULL;
    //fprintf(stdout, "Move constructor used.\n");
}


template <typename T>
cuCV::Mat<T>::~Mat() {
    if (mData != NULL) {
        free(mData);
        mData = NULL;
    }
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator=(Mat mat) { 
    /** This post 
     * https://stackoverflow.com/questions/3106110/what-is-move-semantics
     * recommends to pass mat by value instead of ref, so that the compiler 
     * can choose to take copy or move constructor based on wether the argument 
     * to the assignemnt operator is lvalue or rvalue
    */
    std::swap(mData, mat.mData);

    cuType = mat.cuType;

    mWidth = mat.mWidth; 
    mHeight = mat.mHeight; 
    mStride = mat.mStride;
    mChannels = mat.mChannels;

    return * this;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator+=(const Mat & mat) {
    /// Check dimensions
    if (mWidth != mat.mWidth || mHeight != mat.mHeight || mChannels != mat.mChannels) {
        fprintf(stderr, "Addition failed. Dimensions do not match!");
        exit(EXIT_FAILURE);
    }
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] + (T) mat.mData[i];
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator+=(T alpha) {
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] + alpha;
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator+(const Mat & mat) const {
    Mat<T> sum(*this);  // Copy `this`
    sum += mat;
    return sum;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator+(T alpha) const {
    Mat<T> sum(*this);  // Copy `this`
    sum += alpha;
    return sum;
}



template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator-=(const Mat & mat) {
    /// Check dimensions
    if (mWidth != mat.mWidth || mHeight != mat.mHeight || mChannels != mat.mChannels) {
        fprintf(stderr, "Subtraction failed. Dimensions do not match!");
        exit(EXIT_FAILURE);
    }
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] - (T) mat.mData[i];
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator-=(T alpha) {
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] - alpha;
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator-(const Mat & mat) const {
    Mat<T> sum(*this);  // Copy `this`
    sum -= mat;
    return sum;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator-(T alpha) const {
    Mat<T> sum(*this);  // Copy `this`
    sum -= alpha;
    return sum;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator*=(const Mat & mat) {
    /// Check dimensions
    if (mWidth != mat.mWidth || mHeight != mat.mHeight || mChannels != mat.mChannels) {
        fprintf(stderr, "Subtraction failed. Dimensions do not match!");
        exit(EXIT_FAILURE);
    }
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] * (T) mat.mData[i];
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator*=(T alpha) {
    /// @todo Check if mat is empty

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] * alpha;
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator*(const Mat & mat) const {
    Mat<T> sum(*this);  // Copy `this`
    sum *= mat;
    return sum;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator*(T alpha) const {
    Mat<T> sum(*this);  // Copy `this`
    sum *= alpha;
    return sum;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator/=(const Mat & mat) {
    /// Check dimensions
    if (mWidth != mat.mWidth || mHeight != mat.mHeight || mChannels != mat.mChannels) {
        fprintf(stderr, "Division failed. Dimensions do not match!");
        exit(EXIT_FAILURE);
    }
    /// @todo Check if mat is empty
    /// @todo Check divide by zero

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] / (T) mat.mData[i];
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> & cuCV::Mat<T>::operator/=(T alpha) {
    /// @todo Check if mat is empty
    /// @todo Check divide by zero

    /// Perform Math
    for (int i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = mData[i] / alpha;
    }
    return * this;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator/(const Mat & mat) const {
    Mat<T> sum(*this);  // Copy `this`
    sum /= mat;
    return sum;
}


template <typename T>
cuCV::Mat<T> cuCV::Mat<T>::operator/(T alpha) const {
    Mat<T> sum(*this);  // Copy `this`
    sum /= alpha;
    return sum;
}


template <typename T>
void cuCV::Mat<T>::alloc() {
    if ((mWidth == 0) || (mHeight == 0) || (mChannels == 0)) {
        fprintf(stderr, "You are trying to allocate memory for a mat object but dimensions are: (%d, %d, %d). (FILE: %s), (LINE: %d)\n", mHeight, mWidth, mChannels, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (mData == NULL)
        mData = (T *) malloc(mWidth * mHeight * mChannels * sizeof(T));
    else {
        fprintf(stderr, "mData was not NULL before allocation. mData must be freed before Reallocation. (FILE: %s), (LINE: %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}


template <typename T>
void cuCV::Mat<T>::clear() {
    if (mData != NULL) {
        free(mData);
        mData = NULL;
    }
}


template <typename T>
T cuCV::Mat<T>::at(const int row, const int col) const {
    /// row major: a11, a12, a13, a21, ...
    return mData[row * mStride + col];
}


template <typename T>
T cuCV::Mat<T>::at(const int row, const int col, const int channel) const {
    /// row major: a11, a12, a13, a21, ...
    return mData[row * mStride + col + channel*mWidth*mHeight];
}


template <typename T>
void cuCV::Mat<T>::zeros() {

    if (mData == NULL) { // allocate
        mData = (T *) malloc(mWidth * mHeight * mChannels * sizeof(T));
    }

    for (size_t i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = (T) 0;
        int test = mData[i];
    }
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
void cuCV::Mat<T>::eye() {

    if (mData == NULL) { // allocate
        mData = (T *) malloc(mWidth * mHeight * mChannels * sizeof(T));
    }

    const int STRIDE = mWidth;

    int c = 0;

    // eye matrix without if-else
    for (size_t l = 0; l < mChannels; ++l) {
        size_t i = mWidth * mHeight * l;  // will be index for every entry.
        size_t ii = i; // will be index for diagonal entry only

        for ( ; i < mWidth * mHeight * (l+1); ) {
            ii = i;  // Diagonal entry
            mData[ii] = (T) 1; 
            
            for (i++; (i < ii + STRIDE + 1) && (i < mHeight * mWidth * (l+1)); ++i)
                mData[i] = (T) 0; 
        }
    }
}


template <typename T>
void cuCV::Mat<T>::print(int nRows, int nCols, int channel) const {
    if (mData == NULL) {
        fprintf(stderr, "Data Pointer is NULL.");
    }

    if (nRows > mHeight)
        nRows = mHeight;
        
    if (nCols > mWidth)
        nCols = mWidth;

    for (int row=0; row < nRows; row++) {
        for (int col=0; col < nCols; col++) {
            printf("%f \t", (double) at(row, col, channel));
        }
        printf("\n");
    }
}



/// Explicit template specialization
template class cuCV::Mat<unsigned char>;
template class cuCV::Mat<unsigned short>;
template class cuCV::Mat<double>;
