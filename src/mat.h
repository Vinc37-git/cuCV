/**
 * @file mat.h
 * @author Vincent Hackstein (vin37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <utility>

/**
 * @brief CUCV specific datatypes
 */
#define CUCV_8U unsigned char
#define CUCV_16U unsigned short
#define CUCV_64F double

#define CUCV_8U_MAX 255
#define CUCV_16U_MAX 65535



namespace cuCV {

enum class CuType {cuCV_8U, cuCV_16U, cuCV_64F};

/**
 * @brief The Mat class represents a mathematical matrix with a maximum of 3 dimensions in cuCV.
 * It works as a base class for the cuCV cuMat Matrix Class which can be used for GPU applications. 
 * Member variables are dimension parameters and a pointer pointing to the data of the matrix
 * which must be stored in row major order. Channel data is stored subsequently.
 * Member functions are fundamental mathematical operations (on CPU) as well as initilaizers and some other
 * helpful basic functions.
 * 
 */
template <typename T>
class Mat {
public:
    /**
     * @brief Construct a new empty Mat object. Using this constructor 
     * will require setting of dimensions and/or data pointer in a subsequent step.
     */
    Mat();

    /**
     * @brief Construct a new Mat object pointing to data. Provide dimension parameteres and a pointer to data.
     * Note that data must be stored ROW MAJOR and channels as subsequent data blocks.
     * Use this constructor, when you have data already stored in memory.
     * 
     * @param width The number of columns of the matrix.
     * @param height The number of rows of the matrix.
     * @param channels The number of channels of the matrix.
     * @param data The pointer to the data stored in row major order.
     */
    Mat(int width, int height, int channels, T * data); 

    /**
     * @brief Construct a new empty Mat object. Provide dimension parameteres, but the data pointer will be a `NULL` pointer.
     * Use this constructor, when you want to use a initializer method of the Mat class.
     * 
     * @param width The number of columns of the matrix.
     * @param height The number of rows of the matrix.
     * @param channels The number of channels of the matrix.
     */
    Mat(int width, int height, int channels); 

    /**
     * @brief Construct a new Mat object by copying another mat object.
     * 
     * @param mat 
     */
    Mat(const Mat & mat);

    /**
     * @brief Construct a new Mat object by moving a source mat (Move Constructor).
     * 
     * @param mat 
     */
    Mat(Mat && mat);

    
    /**
     * @brief Destroy the Mat object. Unfreed pointer will be freed. 
     * However, make sure to free all data once it is not needed anymore.
     * NOTE: free() is deactivated to detect leaked memory mistakes.
     * 
     */
    ~Mat();

    
    // ******* OPERATORS **********

    /**
     * @brief Copy Assigment operator.
     * 
     * @param mat 
     * @return Mat& 
     */
    Mat & operator=(Mat mat);

    /**
     * @brief Add one Mat object to the other elementwise. 
     * 
     * @param mat 
     * @return Mat& 
     */
    Mat & operator+=(const Mat & mat);

    /**
     * @brief Add a scalar to a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat& 
     */
    Mat & operator+=(T alpha);

    /**
     * @brief Add on Mat object to another elementwise. Note that `A = A + B` will leak memory, 
     * as the return Mat Object will point to a new chunk of memory and the reference to the 
     * old A Object will be lost.
     * 
     * @param mat 
     * @return Mat 
     */
    Mat operator+(const Mat & mat) const;

    /**
     * @brief Add a scalar to a Mat object elementwise. Note that `A = A + 5` will leak memory, 
     * as the return Mat Object will point to a new chunk of memory and the reference to the old 
     * A Object will be lost.
     * 
     * @param alpha 
     * @return Mat 
     */
    Mat operator+(T alpha) const;

    /**
     * @brief Subtract one Mat object to the other elementwise. 
     * 
     * @param mat 
     * @return Mat& 
     */
    Mat & operator-=(const Mat & mat);

    /**
     * @brief Subtract a scalar to a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat& 
     */
    Mat & operator-=(T alpha);

    /**
     * @brief Subtract on Mat object to another elementwise.
     * 
     * @param mat 
     * @return Mat 
     */
    Mat operator-(const Mat & mat) const;

    /**
     * @brief Subtract a scalar to a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat 
     */
    Mat operator-(T alpha) const;

    /**
     * @brief Multiply one Mat object with the other elementwise. 
     * 
     * @param mat 
     * @return Mat& 
     */
    Mat & operator*=(const Mat & mat);

    /**
     * @brief Multiply a scalar with a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat& 
     */
    Mat & operator*=(T alpha);

    /**
     * @brief Multiply one Mat object with another elementwise.
     * 
     * @param mat 
     * @return Mat 
     */
    Mat operator*(const Mat & mat) const;

    /**
     * @brief Multiply a scalar with a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat 
     */
    Mat operator*(T alpha) const;

    /**
     * @brief Divide one Mat object with the other elementwise. 
     * 
     * @param mat 
     * @return Mat& 
     */
    Mat & operator/=(const Mat & mat);

    /**
     * @brief Divide a scalar with a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat& 
     */
    Mat & operator/=(T alpha);

    /**
     * @brief Divide one Mat object with another elementwise.
     * 
     * @param mat 
     * @return Mat 
     */
    Mat operator/(const Mat & mat) const;

    /**
     * @brief Divide a scalar with a Mat object elementwise.
     * 
     * @param alpha 
     * @return Mat 
     */
    Mat operator/(T alpha) const;


    // ******* PUBLIC SETTERS AND GETTERS *******
    
    /**
     * @brief Get the Width of Mat.
     * 
     * @return The width as int.
     */
    int getWidth() const;

    /**
     * @brief Get the height of Mat.
     * 
     * @return The height as int.
     */
    int getHeight() const;

    /**
     * @brief Get the number of channels of Mat.
     * 
     * @return The number of channels as int.
     */
    int getNChannels() const;

    /**
     * @brief Get the stride of Mat.
     * 
     * @return The stride as int.
     */
    int getStride() const;

    /**
     * @brief Get the data pointer of Mat.
     * 
     * @return The data pointer as pointer to data of type T.
     */
    T * getDataPtr() const;


    //  ******* MEMBER FUNCTIONS **********

    /**
     * @brief Get a specific element of a 2d Matrix (or an element of the first channel of a 3d Matrix).
     * 
     * @param row 
     * @param col 
     * @return The element at the given position. May be of any cuCV datatype.
     */
    T at(const int row, const int col) const;

    /**
     * @brief Get a specific element of a 3d Matrix.
     * 
     * @param row 
     * @param col 
     * @param ch 
     * @return The element at the given position. May be of any cuCV datatype. 
     */
    T at(const int row, const int col, const int ch) const;

    /**
     * @brief Initialize the mat object with zeros.
     */
    void zeros() ;

    /**
     * @brief Initialize the mat object with ones.
     */
    void ones() ;

    /**
     * @brief Initialize the mat object as Identity Matrix.
     */
    void eye() ;

    /**
     * @brief Check if matrix contains data.
     * 
     * @return true if empty, otherwise false.
     */
    bool empty() const;

    /**
     * @brief Print the first number of given rows and columns to the std output stream.
     * 
     * @param nRows The number of rows starting from first.
     * @param nCols The number of cols starting from first.
     * @param channel The channel to be printed. Defaults to 0.
     */
    void print(int nRows, int nCols, int channel=0) const;

    /**
     * @brief Allocate memory for a mat object with a already given size.
     */
    void alloc();   ///< Maybe

    /**
     * @brief Free memory associated with the mat object.
     */
    void clear();  ///< Maybe
    
    /**
     * @brief cuCV Datatypes.
     */
    CuType cuType;

//protected:
    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mStride;  ///< Stride of the matrix represented by the mat object.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.

};

};


#endif // 