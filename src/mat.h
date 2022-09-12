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
#include <stdexcept>

/**
 * @brief CUCV specific datatypes
 */
#define CUCV_8U unsigned char
#define CUCV_16U unsigned short
#define CUCV_32F float
#define CUCV_64F double

#define CUCV_8U_MAX 0xff
#define CUCV_16U_MAX 0xffff

#ifndef CUCV_DEBUG
#define CUCV_DEBUG 0
#endif

/**
 * @brief A debug print macro. It will print debug messages when CUCV_DEBUG is defined at compile time.
 * However, if CUCV_DEBUG is not defined the optimizer should remove the code after compilation.
 * A formated string can be passed as usual when using `fprintf` or `printf`.
 */
#define CUCV_DEBUG_PRINT(str, ...) \
        do { if (CUCV_DEBUG) fprintf(stdout,"[%10s] : " str " (Line: %d, File: %s)\n", __func__, __VA_ARGS__, __LINE__, __FILE__); } while (0)



/**
 * @brief A CUDA accelerated Computer Vision Library.
 * 
 */
namespace cuCV {


/**
 * @brief ENUM CURRENTLY NOT USED. 
 * Available datatypes in CUCV: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * 
 */
enum class CuType {cuCV_8U, cuCV_16U, cuCV_32F, cuCV_64F};


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
     * If borrowed is true, the mat instance will only borrow the data. This means it will not copy or steal the data
     * and hence, not take care about memory management. You must take care when operating on borrowed data, since it
     * might be an invalid buffer (eg already deallocated)
     * If borrowed is set to false, the mat instance will steal the data. This implies it will try to delete / free the
     * data on destruction. You must ensure that no other instance is responsible for data, since it could result in an 
     * invalid free() call or invalid buffer when operating on it.
     * 
     * @param width The number of columns of the matrix.
     * @param height The number of rows of the matrix.
     * @param channels The number of channels of the matrix.
     * @param data The pointer to the data stored in row major order.
     * @param borrowed Indicates if data is borrowed only or will be stealed. Defaults to true.
     */
    Mat(int width, int height, int channels, T * data, bool borrowed=true); 

    /**
     * @brief Construct a new empty Mat object. Provide dimension parameteres, but the data pointer will be a `NULL` pointer.
     * Use this constructor, when you want to use an initializer method of the Mat class.
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
     * @brief Destroy the Mat object. Allocated memory will be freed automatically,
     * if the data is not borrowed. If the data is borrowed, nothing will happen. <br>
     * However, consider to clear the matrix by yourself directly if you do not need it anymore
     * or let it go out of scope so that the destructor frees all data.
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
     * @brief Add on Mat object to another elementwise.
     * 
     * @param mat 
     * @return Mat 
     */
    Mat operator+(const Mat & mat) const;

    /**
     * @brief Add a scalar to a Mat object elementwise.
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
    
    /** @brief Get the width of Mat. */
    int getWidth() const;
    
    /** @brief Set the width of Mat. @deprecated: use initShape or reshape. */
    // void setWidth(int width);

    /** @brief Get the height of Mat. */
    int getHeight() const;
    
    /** @brief Set the height of Mat. @deprecated: use initShape or reshape. */
    // void setHeight(int height);

    /** @brief Get the depth of Mat. */
    int getNChannels() const;
    
    /** @brief Set the depth of Mat. @deprecated: use initShape or reshape. */
    // void setNChannels(int channels);

    /** @brief Get the Stride in width-direction of Mat. */
    int getStrideX() const;
    
    /** @brief Set the Stride in width-direction of Mat. @deprecated: use initShape or reshape. */
    // void setStrideX(int strideX);

    /** @brief Get the Stride in height-direction of Mat. */
    int getStrideY() const;

    /** @brief Set the Stride in height-direction of Mat. @deprecated: use initShape or reshape. */
    // void setStrideY(int strideY);

    /** @brief Get thenumber of elements of Mat. */
    size_t getSize() const;

    /** @brief Get the Data pointer pointing to the first element of Mat. */
    T * getDataPtr() const;
    
    /** @brief Set the Data pointer pointing to the first element of Mat. 
     * Note that the user must guarantee the size of the array pData is pointing to
     * matches the number of elements in the mat object.
    */
    void setDataPtr(T * pData);

    /**
     * @brief Initialize the shape of a matrix, in case it was not initialized yet 
     * (if the standard constructor was used before). This works if all dimension
     * parameters are set to 0 only. This prevents that the mat will have an invalid
     * number of elements that does not match the number of elements in the data
     * array. This method replaces the standard setters of the dimension parameters.
     * @todo add reshape method.
     * 
     * @param width 
     * @param height 
     * @param channels 
     * @param strideX 
     * @param strideY 
     */
    void initShape(int width, int height, int channels, int strideX=-1, int strideY=-1);


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

    /**  @brief Initialize the mat object with zeros. */
    void zeros() ;

    /** @brief Initialize the mat object with ones. */
    void ones() ;

    /** @brief Initialize the mat object as Identity Matrix. */
    void eye() ;

    /**
     * @brief Return a copy of the mat object as type dtype. 
     * Note that the user has to take care about overflows before conversion.
     * 
     * @tparam Tout The target data type.
     * @return A copy of the mat object as type dtype.
     */
    template <typename Tout>
    Mat<Tout> astype();

    /** @brief Check if matrix contains data. Return true if empty, otherwise false. */
    bool empty() const;

    /**
     * @brief Print the first number of given rows and columns to the std output stream.
     * 
     * @param nRows The number of rows starting from first.
     * @param nCols The number of cols starting from first.
     * @param channel The channel to be printed. Defaults to 0.
     */
    void print(int nRows, int nCols, int channel=0) const;

    /** @brief Allocate memory for a mat object with a already given size. */
    void alloc();

    /** @brief Free memory associated with the mat object. */
    void clear();  ///< Maybe
    
    /** @brief @note CURRENTLY NOT USED. cuCV Datatypes.  */
    CuType cuType;


protected:
    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    int mStrideX;  ///< Stride of the memory in x direction.
    int mStrideY;  ///< Stride of the memory in y direction.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.
    bool mBorrowed; ///< Indicates if data of mat is borrowed only. If borrowed, it will not try to deallocate on destruction.
};

};


#endif // 