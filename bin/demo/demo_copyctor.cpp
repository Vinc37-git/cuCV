#include <mat.h>
#include <initializers.h>
int main() {
    // Scenario, if no copy-constructor and copy-assignment operator was defined,
    // but destructor automatically deletes its data.
    cuCV::Mat<CUCV_32F> A = cuCV::ones<CUCV_32F>(128,64,3);
    {
        cuCV::Mat<CUCV_32F> B = A;  // Would not copy data of A, but only pointer
    }  // destroys B. Frees data of B (and A)
    A += 1;  // Would raise cuCV::exception::NullPointer
}