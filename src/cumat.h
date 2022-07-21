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
#include "kernelcumat.h"

#define BLOCK_SIZE 32


namespace cuCV {

/**
 * @brief The CuMat class is inherited from the basic Mat class and represents a mathematical matrix 
 * with a maximum of 3 dimensions on a cuda device in cuCV. It can be used to perform mathematical operations
 * on the device using cuda kernels. Public and protected member variables are inherited from the base class 
 * and provide informations about the dimensions of the matrix. Note that the developer must take care of both 
 * data transfer AND data deletion, meaning data is not automatically deleted from the device after the download to the host.
 * This allows data reading on the Host while it can still be used for further calculations on device. However, data Allocation
 * on device is usually done before data transfer automatically.
 * 
 * @tparam T 
 */
template <typename T>
class CuMat : public Mat<T> {
public:
    /**
     * @brief Construct a new CuMat object using the standard constructor. Using this constructor 
     * will require setting of dimensions and/or data pointer in a subsequent step.
     */
    CuMat();


    /**
     * @brief Construct a new CuMat object from a Mat Object on the host. Dimension parameters will be assigned
     * automatically. However, no data will be allocated on the device. 
     * @param mat 
     */
    CuMat(Mat<T> & mat);


    CuMat(int width, int height, int channels);


    /**
     * @brief Construct a new CuMat object by copying another CuMat Object on the device.
     * 
     * @param cuMat 
     */
    CuMat(const CuMat & cuMat);


    /**
     * @brief Construct a new CuMat object by moving a source CuMat Object on the device.
     * 
     * @param cuMat 
     */
    CuMat(CuMat && cuMat);
    
    
    //CuMat(int width, int height, int channels, void * data);


    /**
     * @brief Destroy the Cu Mat object. If the object points to memory on device, this data will be freed.
     */
    ~CuMat();


    // ******* OPERATORS **********

    /**
     * @brief Copy Assigment operator.
     * 
     * @param cuMat 
     * @return a copy of cuMat
     */
    CuMat & operator=(CuMat cuMat);


    /**
     * @brief Add one CuMat object to the other elementwise and assign it to itself. 
     * 
     * @param cuMat 
     * @return CuMat& 
     */
    CuMat & operator+=(const CuMat & cuMat);
    CuMat & operator+=(const T alpha);


    /**
     * @brief Add one CuMat object to the other elementwise.
     * 
     * @param cuMat 
     * @return CuMat 
     */
    CuMat operator+(const CuMat & cuMat) const;
    CuMat operator+(const T alpha) const;


    /**
     * @brief Subtract one CuMat object to the other elementwise and assign it to itself. 
     * 
     * @param cuMat 
     * @return CuMat& 
     */
    CuMat & operator-=(const CuMat & cuMat);
    CuMat & operator-=(const T alpha);


    /**
     * @brief Subtract one CuMat object to the other elementwise.
     * 
     * @param cuMat 
     * @return CuMat 
     */
    CuMat operator-(const CuMat & cuMat) const;
    CuMat operator-(const T alpha) const;


    /**
     * @brief Multiply one CuMat object with the other elementwise and assign it to itself. 
     * 
     * @param cuMat 
     * @return CuMat& 
     */
    CuMat & operator*=(const CuMat & cuMat);
    CuMat & operator*=(const T alpha);


    /**
     * @brief Multiply one CuMat object to the other elementwise.
     * 
     * @param cuMat 
     * @return CuMat 
     */
    CuMat operator*(const CuMat & cuMat) const;
    CuMat operator*(const T alpha) const;


    /**
     * @brief Divide one CuMat object with the other elementwise and assign it to itself. 
     * 
     * @param cuMat 
     * @return CuMat& 
     */
    CuMat & operator/=(const CuMat & cuMat);
    CuMat & operator/=(const T alpha);


    /**
     * @brief Divide one CuMat object with the other elementwise.
     * 
     * @param cuMat 
     * @return CuMat 
     */
    CuMat operator/(const CuMat & cuMat) const;
    CuMat operator/(const T alpha) const;


    // ******* DATA MANAGEMENT METHODS **********


    void uploadFrom(const Mat<T> & srcMat);
    void downloadTo(Mat<T> & dstMat) const;

    void allocateOnDevice();
    void allocateLike(const Mat<T> & srcMat);
    void clearOnDevice();

    KernelCuMat<T> kernel() const;


//private:
    /**
     * @brief Compare Dimenions of Matrix A and B on Equality.
     * 
     * @param A Matrix A
     * @param B Matrix B
     * @return true if dimensions are equal. Otherwise false.
     */
    bool compareDim(const CuMat & A, const CuMat & B) const;
    bool compareDim(const CuMat & A, const Mat<T> & B) const;
    bool compareDim(const Mat<T> & A, const CuMat & B) const;
    bool compareDim(const Mat<T> & A, const Mat<T> & B) const;


    /**
     * @brief Check if Matrix points to data.
     * 
     * @return true if data pointer is null.
     */
    bool empty() const;
};

};


 #endif //