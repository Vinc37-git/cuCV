/**
 * @file Mat.h
 * @author Vincent Hackstein (vin37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __Mat.h__
#define __Mat.h__

namespace cuCV {

    /**
     * @brief 
     * 
     */
    class Mat {
        private:
            int width;
            int height;
            int stride;

            char * elements; 
            int * elements;  ///< @todo Search for a nicer method.
            float * elements;

            char * dev_elements;

            void sent();

            void download();c++

        public:
            Mat(int width, int height, int channels, char * data);

    };

}


#endif // 