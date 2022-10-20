/**
 * @file cucvconv.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include <filesystem>

#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "linalg.h"
#include "imread.h"

void usage() {
    const char * usage = 
            "Usage: cucvconv [OPTIONS]... [FILE]...\n"
            "Filter an image at path FILE with a selection of predefined filters\n"
            "and filter sizes. Filtered images will written with a _filt suffix.\n"
            "\n"
            "Options are:\n"
            "-k \t The kernel to filter the image. It must be a numeric value: \n"
            "   \t 0: BOX, 1:unnormalized BOX, 2: SobelX, 3: SobelY, 4: Laplace, 5: Gauss\n"
            "   \t Default is Gauss.\n"
            "-s \t Kernel Length of one side. Default is 3.\n"
            "\n"
            "PLEASE NOTE THAT THE FUNCTION IS EXPERIMENTAL AND MIGHT NOT WORK IF NOT USED AS INTENDED.\n";
    printf("%s", usage);
}

bool parseCmdLine(int argc, char ** argv, int & k, int & s) {
    if (argc <= 1) {
        usage();
        fprintf(stderr, "\nError: No Filepath provided.\n");
        return false;
    }
    if (argc % 2 != 0) {  // an option must come with a value.
        usage();
        fprintf(stderr, "\nError: Please provide a value with your option.\n");
        return false;
    }
    for (int i=1; i<argc; i=i+2) {
        if (i+1 < argc) {
            if (strcmp(argv[i], "-k") == 0) {
                k = atoi(argv[i+1]);
                if (k < 0 || k > 5) {
                    usage();
                    fprintf(stderr, "\nError: Spcified Kernel not available.\n");
                    return false;
                }
            }
            if (strcmp(argv[i], "-s") == 0) {
                s = atoi(argv[i+1]);
                if (s < 0) {
                    usage();
                    fprintf(stderr, "\nError: Spcified Kernel Size not allowed.\n");
                    return false;
                }
            }
        }
    }
    return true;
}


int main(int argc, char ** argv) {

    const char * path = NULL;
    int k = (int) cuCV::Kernel::GAUSS;
    int s = 3;

    if (!parseCmdLine(argc, argv, k, s))
        return EXIT_FAILURE;

    path = argv[argc-1];
    printf("Reading image at %s.\n", path);

    // Load Image to device
    cuCV::CuMat image = cuCV::imreadToDevice(path);  

    // Create kernel
    cuCV::CuMat kernel = cuCV::createKernel((cuCV::Kernel) k, s, s, 1);  

    // Convolve
    cuCV::CuMat<CUCV_8U> out = cuCV::simpleSharedConv2d(image, kernel, cuCV::Padding::ZERO);
    
    // write
    std::filesystem::path fs_path = path;
    std::filesystem::path file = fs_path.stem();
    std::filesystem::path ext  = fs_path.extension();
    file += "_filt";
    file += ext;
    std::filesystem::path new_path = fs_path.parent_path() / file;

    printf("Writing image to %s.\n", new_path.c_str());
    cuCV::imwrite(out, new_path.c_str());
    
    return EXIT_SUCCESS;
}