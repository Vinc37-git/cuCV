/**
 * @file cumat.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef CUMAT_H
#define CUMAT_H



#include <iostream>
#include <unistd.h>


#include "errorhandling.h"
#include "mat.h"
#include "kernel.h"

#define BLOCK_SIZE 5


namespace cuCV {

template <typename T>
class CuMat : public Mat<T> {
public:

    CuMat();
    CuMat(Mat<T> & SRC);
    //CuMat(int width, int height, int channels, void * data);

    void add(CuMat & OUT, CuMat & B);

    void uploadFrom(const Mat<T> & srcMat);
    void downloadTo(const Mat<T> & dstMat) const;

private:
    bool allocateLike(const Mat<T> & srcMat);
    void free();

};

};


 #endif //