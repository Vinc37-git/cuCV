#include <mat.h>
#include <initializers.h>

int main() {
    cuCV::Mat<CUCV_32F> A, B, C;
    B = cuCV::ones<CUCV_32F>(128,64,3);

    printf("A = B\n");
    A = B;  // Copy Construktor will be used.
    printf("C = A + B\n");
    C = A + B;  // Move Construktor will steal data of rvalue expression A+B. 
}