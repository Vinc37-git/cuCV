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

#define CUCV_8U unsigned char
#define CUCV_16U unsigned short
#define CUCV_64F double


namespace cuCV {

enum class CuType {cuCV_8U, cuCV_16U, cuCV_64F};

/**
 * @brief 
 * 
 */
template <typename T>
class Mat {
public:
    Mat();
    Mat(int width, int height, int channels, T * data); 
    Mat(int width, int height, int channels); 

    Mat operator+(const Mat & mat) const;
    Mat operator+(T alpha) const;

    T at(const int row, const int col) const;
    T at(const int row, const int col, const int ch) const;
    void ones() ;
    void eye() ;
    bool empty() const;
    void print(int nRows, int nCols) const;

//protected:
    void alloc();   ///< Maybe
    void free();  ///< Maybe
    
    CuType cuType;

    int mWidth;
    int mHeight;
    int mStride;
    int mChannels;
    T * mData;
    //bool onDevice;

};

};


#endif // 