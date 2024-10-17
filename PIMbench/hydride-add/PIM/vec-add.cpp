// Test: C++ version of vector addition
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

using namespace std;


int main(){
    pimCreateDevice(PIM_FUNCTIONAL, /* Rank*/  4, /* Banks Per Rank */ 128,/*SubArray Per Bank*/ 32, /* NumRows */ 1024,/* Num Cols*/  1024);

    int32_t Operand0[32];
    int32_t Operand1[32];
    int32_t Dst[32];

    for(int i =0; i < 32; i++){
        Operand0[i] = 1;
    } 
    for(int i =0; i < 32; i++){
        Operand1[i] = 1;
    } 
    for(int i =0; i < 32; i++){
        Dst[i] = 0;
    } 

    for(int j = 0; j < 1000; j++){
        auto pimAlloc0 = pimAlloc(PIM_ALLOC_AUTO, 32, PIM_INT32);
        pimCopyHostToDevice((void*) Operand0, pimAlloc0);

        auto pimAlloc1 = pimAllocAssociated(pimAlloc0,PIM_INT32 );
        pimCopyHostToDevice((void*) Operand1, pimAlloc1);

        auto pimAllocDst = pimAllocAssociated(pimAlloc0, PIM_INT32);

        auto AddInst = pimAdd(pimAlloc0, pimAlloc1, pimAllocDst);

        auto CopyResult = pimCopyDeviceToHost(pimAllocDst, (void*) Dst);

        pimFree(pimAlloc0);
        pimFree(pimAlloc1);
        pimFree(pimAllocDst);
    }

    pimShowStats();

    for(int i =0; i < 32; i++){
        if(Dst[i] != 2){
            std::cout<< "Incorrect value at index: "<< i << "\n";
        }
    } 
    return 0;
}

