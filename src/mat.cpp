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
        : mWidth(0), mHeight(0), mChannels(0), 
        mStrideX(0), mStrideY(0), mData(NULL),
        mBorrowed(false) { }


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels, T * data, bool isBorrowed) 
        : mWidth(width), mHeight(height), mChannels(channels), 
        mStrideX(width), mStrideY(height), mData(data),
        mBorrowed(isBorrowed) {
    if (mData == NULL && isBorrowed)
        fprintf(stderr, "Warning: Mat constructed pointing to NULL, but data is set as 'borrowed'.");
}


template <typename T>
cuCV::Mat<T>::Mat(int width, int height, int channels) 
        : mWidth(width), mHeight(height), mChannels(channels), 
        mStrideX(width), mStrideY(height), mData(NULL),
        mBorrowed(false) {
    // No Allocation yet. Allocation will be done when the image is invoked at the first time.
}


template <typename T>
cuCV::Mat<T>::Mat(const Mat & mat) 
        : mWidth(mat.mWidth), mHeight(mat.mHeight), mChannels(mat.mChannels), 
        mStrideX(mat.mStrideX), mStrideY(mat.mStrideY), mData(NULL),
        mBorrowed(false) {
    if (mat.mData != NULL) {
        mData = new T [mWidth * mHeight * mChannels];
        memcpy(mData, mat.mData, mWidth * mHeight * mChannels * sizeof(T));
        CUCV_DEBUG_PRINT("Copy: %p copied to %p.", mat.mData, mData);
    }
}


template <typename T>
cuCV::Mat<T>::Mat(Mat && mat) 
        : mWidth(mat.mWidth), mHeight(mat.mHeight), mChannels(mat.mChannels), 
        mStrideX(mat.mStrideX), mStrideY(mat.mStrideY), mData(mat.mData),
        mBorrowed(mat.mBorrowed) {
    CUCV_DEBUG_PRINT("Move: %p swaped with %p.", mat.mData, mData);
    mat.mData = NULL;
}


template <typename T>
cuCV::Mat<T>::~Mat() {
    if (mData != NULL && !mBorrowed) {
        CUCV_DEBUG_PRINT("%p destroyed.", mData);
        delete [] mData;
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
    mStrideX = mat.mStrideX;
    mStrideY = mat.mStrideY;
    mChannels = mat.mChannels;
    mBorrowed = mat.mBorrowed;

    CUCV_DEBUG_PRINT("%p swaped with %p.", mat.mData, mData);

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
    /// @todo Throw exception

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
    /// @todo Throw exception

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
    /// @todo Throw exception

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
    /// @todo Throw exception

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
int cuCV::Mat<T>::getWidth() const {
    return mWidth;
}


template <typename T>
int cuCV::Mat<T>::getHeight() const {
    return mHeight;
}


template <typename T>
int cuCV::Mat<T>::getNChannels() const {
    return mChannels;
}


template <typename T>
int cuCV::Mat<T>::getStrideX() const {
    return mStrideX;
}


template <typename T>
int cuCV::Mat<T>::getStrideY() const {
    return mStrideY;
}


template <typename T>
T * cuCV::Mat<T>::getDataPtr() const {
    return mData;
}


template <typename T>
void cuCV::Mat<T>::setDataPtr(T * pData) {
    mData = pData;
}


template <typename T>
size_t cuCV::Mat<T>::getSize() const {
    return mWidth * mHeight * mChannels;
}


template <typename T>
void cuCV::Mat<T>::initShape(int width, int height, int channels, int strideX, int strideY) {
    if (mWidth == 0 && mHeight == 0 && mChannels == 0 && mStrideX == 0 && mStrideY == 0) {
        mWidth = width;
        mHeight = height;
        mChannels = channels;
        mStrideX = (strideX == -1) ? width : strideX;
        mStrideY = (strideY == -1) ? height : strideY;
    }
    else
        throw std::runtime_error("Re-Initialization of the shape of a matrix object is not allowed. "
                    "Use reshape() method instead.");
}


template <typename T>
void cuCV::Mat<T>::alloc() {
    if ((mWidth == 0) || (mHeight == 0) || (mChannels == 0)) {
        fprintf(stderr, "You are trying to allocate memory for a mat object but dimensions are: " 
                "(%d, %d, %d). (FILE: %s), (LINE: %d)\n", mHeight, mWidth, mChannels, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (mData == NULL)
        mData = new T [mWidth * mHeight * mChannels];
    else {
        fprintf(stderr, "mData was not NULL before allocation. mData must be freed before Reallocation. (FILE: %s), (LINE: %d)\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    CUCV_DEBUG_PRINT("Allocated %ld bytes at %p.", getSize() * sizeof(T), mData);
}


template <typename T>
void cuCV::Mat<T>::clear() {
    if (mData != NULL && !mBorrowed) {
        delete [] mData;
        mData = NULL;
    }
}


template <typename T>
T cuCV::Mat<T>::at(const int row, const int col) const {
    /// row major: a11, a12, a13, a21, ...
    return mData[row * mStrideX + col];
}


template <typename T>
T cuCV::Mat<T>::at(const int row, const int col, const int channel) const {
    /// row major: a11, a12, a13, a21, ...
    return mData[row * mStrideX + col + channel * mStrideX * mStrideY];
}


template <typename T>
void cuCV::Mat<T>::zeros() {

    alloc();

    for (size_t i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = (T) 0;
        int test = mData[i];
    }
}


template <typename T>
void cuCV::Mat<T>::ones() {

    alloc();

    for (size_t i=0; i < mWidth * mHeight * mChannels; i++) {
        mData[i] = (T) 1;
    }
}


template <typename T>
void cuCV::Mat<T>::eye() {

    alloc();

    const int STRIDE = mStrideX;

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
template <typename Tout>
cuCV::Mat<Tout> cuCV::Mat<T>::astype() {

    /** @todo: check if (type == this->type) return copy */

    cuCV::Mat<Tout> OUT(mWidth, mHeight, mChannels);
    OUT.alloc();
    for (size_t i = 0; i < OUT.getSize(); i++)
        OUT.getDataPtr()[i] = static_cast<Tout>(mData[i]);
    return OUT;
}


template <typename T>
void cuCV::Mat<T>::print(int nRows, int nCols, int channel) const {
    if (mData == NULL) {
        fprintf(stderr, "print() method failed: Data Pointer is NULL.");
        return;
    }

    if (nRows > mHeight)
        nRows = mHeight;
        
    if (nCols > mWidth)
        nCols = mWidth;

    printf("  ");
    for (int col=0; col < nCols; col++)
        printf("%3d ", col);
    printf("\n");

    for (int row=0; row < nRows; row++) {
        printf("%d : ", row);
        for (int col=0; col < nCols; col++) {
            printf("%.1f ", (double) at(row, col, channel));
        }
        printf("\n");
    }
}



/// Explicit template specialization
template class cuCV::Mat<unsigned char>;
template class cuCV::Mat<unsigned short>;
template class cuCV::Mat<CUCV_32F>;
template class cuCV::Mat<double>;

template cuCV::Mat<CUCV_8U> cuCV::Mat<CUCV_8U>::astype();
template cuCV::Mat<CUCV_16U> cuCV::Mat<CUCV_8U>::astype();
template cuCV::Mat<CUCV_32F> cuCV::Mat<CUCV_8U>::astype();
template cuCV::Mat<CUCV_64F> cuCV::Mat<CUCV_8U>::astype();
template cuCV::Mat<CUCV_8U> cuCV::Mat<CUCV_16U>::astype();
template cuCV::Mat<CUCV_16U> cuCV::Mat<CUCV_16U>::astype();
template cuCV::Mat<CUCV_32F> cuCV::Mat<CUCV_16U>::astype();
template cuCV::Mat<CUCV_64F> cuCV::Mat<CUCV_16U>::astype();
template cuCV::Mat<CUCV_8U> cuCV::Mat<CUCV_32F>::astype();
template cuCV::Mat<CUCV_16U> cuCV::Mat<CUCV_32F>::astype();
template cuCV::Mat<CUCV_32F> cuCV::Mat<CUCV_32F>::astype();
template cuCV::Mat<CUCV_64F> cuCV::Mat<CUCV_32F>::astype();
template cuCV::Mat<CUCV_8U> cuCV::Mat<CUCV_64F>::astype();
template cuCV::Mat<CUCV_16U> cuCV::Mat<CUCV_64F>::astype();
template cuCV::Mat<CUCV_32F> cuCV::Mat<CUCV_64F>::astype();
template cuCV::Mat<CUCV_64F> cuCV::Mat<CUCV_64F>::astype();
