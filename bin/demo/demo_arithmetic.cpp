#include <mat.h>
#include <initializers.h>

int main() {
    cuCV::Mat<CUCV_32F> A = cuCV::ones<CUCV_32F>(128,64,3);
    cuCV::Mat<CUCV_32F> B = A * 25 / 7;
    cuCV::Mat<CUCV_32F> C = A + B - A * 2;
    
    C.print(1,1);  // will print: 
    C.astype<CUCV_16U>().print(1,1);  // will truncate C and print
    return EXIT_SUCCESS;
}