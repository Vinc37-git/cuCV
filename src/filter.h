/**
 * @file filter.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef FILTER_H
#define FILTER_H


namespace cuCV {

enum class Padding {/*NONE,*/ ZERO/*, SAME*/};

enum class Kernel {BOX, SOBELX, SOBELY, LAPLACE, GAUSS};
    
}  // namespace cuCV

#endif  // FILTER_H