#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include "errorhandling.h"

int main() {
    // Get the top level suite from the registry
    CPPUNIT_NS::Test *suite = CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest();

    CppUnit::TextUi::TestRunner runner;
    runner.addTest(suite);
    bool wasSucessful = runner.run();

    #if CUDA_MEMCHECK_ABLE
        fprintf(stdout, "Build with cudaDeviceReset()\n");
        gpuErrchk(cudaDeviceReset());  // to detect leaks with cuda-memcheck we need to call cudaDeviceReset(). 
    #endif

    return wasSucessful ? 0 : 1;
}