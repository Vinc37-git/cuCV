/**
 * @file cucvconv.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief A tool to read an image and convolve it using a set of filters.
 * Choose Filter and Filter size using command line tools
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

/** @brief Usage printout. */
static void usage() {
    const char * usage = 
            "Usage: cucvconv [OPTIONS]... [FILE]...\n"
            "Filter an image at path FILE with a selection of predefined filters\n"
            "and filter sizes. Filtered images will written with a _filt suffix.\n"
            "\n"
            "Options are:\n"
            "-h, --help \t Display this message.\n"
            "-k, --kernel \t The kernel to filter the image. It must be a numeric value: \n"
            "   \t\t 0: BOX, 1:unnormalized BOX, 2: SobelX, 3: SobelY, 4: Laplace, 5: Gauss\n"
            "   \t\t Default is Gauss.\n"
            "-s, --size \t Kernel Length of one side. Default is 3.\n"
            "\n"
            "PLEASE NOTE THAT THE FUNCTION IS EXPERIMENTAL AND MIGHT NOT WORK IF NOT USED AS INTENDED.\n";
    printf("%s", usage);
}


/**
 * @brief Parse a command line for cucvconv
 * 
 * @param argc argc of main
 * @param argv argv of main
 * @param k kernel choice
 * @param s kernel size
 * @return false if abort
 */
static bool parseCmdLine(int argc, char ** argv, int & k, int & s) {
    if (argc <= 1) {  // No filepath?
        usage();
        fprintf(stderr, "\nError: No Filepath provided.\n");
        return false;
    }
    for (int i=1; i<argc; ++i) {  // help argument
        if (strcmp(argv[i], "-h") == 0 || (strcmp(argv[i], "--help") == 0)) {
            usage();
            return false;
        }
    }
    if (argc % 2 != 0) {  // an option must come with a value.
        usage();
        fprintf(stderr, "\nError: Please provide a [FILE] path and value with every [OPTIONS].\n");
        return false;
    }
    for (int i=1; i<argc; i=i+2) {  // Loop over option-value pairs
        if (i+1 < argc) {
            if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0) {
                k = atoi(argv[i+1]);
                if (k < 0 || k > 5) {
                    usage();
                    fprintf(stderr, "\nError: Spcified Kernel not available.\n");
                    return false;
                }
            }
            else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--size") == 0) {
                s = atoi(argv[i+1]);
                if (s < 1) {
                    usage();
                    fprintf(stderr, "\nError: Spcified Kernel Size %i not allowed.\n", s);
                    return false;
                }
            }
            else {
                usage();
                fprintf(stderr, "\nError: Unknown option-value pair: %s and %s.\n", argv[i], argv[i+1]);
                return false;
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

    /** @todo Convert RGB images to Greyscale before convolution */
    if (image.getNChannels() > 1) {
        fprintf(stderr, "\nError: Currently, only greyscale images are supported.");
        return EXIT_FAILURE;
    }

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