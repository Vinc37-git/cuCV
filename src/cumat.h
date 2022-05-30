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

 #define BLOCK_SIZE 32

#include <iostream>

#include "mat.h"
#include "kernel.h"

namespace cuCV {

template <typename T>
class CuMat : public Mat<T> {
public:

    CuMat();
    CuMat(int width, int height, int channels, void * data); ///< We will use templates later

    void add(CuMat & OUT, CuMat & B);

    void allocate(const Mat<T> & srcMat);
    void free();
    void uploadFrom(const Mat<T> & srcMat);
    void downloadTo(const Mat<T> & dstMat) const;


};

};


 #endif //