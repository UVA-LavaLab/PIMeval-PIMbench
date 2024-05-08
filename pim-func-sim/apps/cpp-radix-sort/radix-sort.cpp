// Radix Sort implementation on bitSIMD
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>

#include <cstdlib>
#include <time.h> 
#include <limits.h>
#include <algorithm>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main()
{
    int SCALE = 1;
    std::cout << "PIM test: Radix Sort" << std::endl;

    unsigned numCores = 8;  //make sure to select the correct number
    unsigned numRows = 128; //make sure this is possible
    unsigned numCols = 256 * SCALE;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return 1;
    }

    unsigned numElements = 512 * SCALE;
    unsigned bitsPerElement = 32;
    unsigned radix_bits = 8;
    unsigned num_passes = bitsPerElement / radix_bits;
    unsigned radix = 1 << radix_bits;

    //Allocating Pimobj for all the iterations
    std::vector<PimObjId> src_obj(num_passes);
    std::vector<PimObjId> compare_obj(num_passes);
    std::vector<PimObjId> compare_results_obj(num_passes);

    //What is the difference between bitsPerElement and PIM_INT32
    for(unsigned i = 0; i < num_passes; i++){
        src_obj[i] = pimAlloc(PIM_ALLOC_V1, numElements, bitsPerElement, PIM_INT32);
        if (src_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }
    //How does the association affect the computation?
    for(unsigned i = 0; i < num_passes; i++){
        compare_obj[i] = pimAllocAssociated(PIM_ALLOC_V1, numElements, bitsPerElement, src_obj[i], PIM_INT32);
        if (compare_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }
    for(unsigned i = 0; i < num_passes; i++){
        compare_results_obj[i] = pimAllocAssociated(PIM_ALLOC_V1, numElements, bitsPerElement, src_obj[i], PIM_INT32);
        if (compare_results_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }

    //vectore for host use
    std::vector<unsigned> src1(numElements);
    std::vector<unsigned> dest(numElements);
    //array used to check result
    std::vector<unsigned> sorted_array(numElements);
    //counting table in host
    std::vector<unsigned> count_table(radix);
    
    srand((unsigned)time(NULL));
    //Assign random initial values to the input array
    for (unsigned i = 0; i < numElements; ++i) {
        src1[i] = rand() % UINT_MAX;
    }
    sorted_array = src1;

    unsigned mask = 0x000000FF;
    // auto duration_cpu = high_resolution_clock::now() - high_resolution_clock::now();//initialize it to be 0

    //Outer iteration of radix sort, each iteration perform a counting sort
    // auto start_total = high_resolution_clock::now();
    for (unsigned i = 0; i < num_passes; i++){
        std::fill(count_table.begin(), count_table.end(), 0);

        //Create a slice of 'radix_bits' of the input array and only copy that array to bitSIMD
        std::vector<unsigned> src1_slice(numElements);  //shoud be an array of 8-bit elements if radix_bits=8
        for (unsigned j = 0; j < numElements; j++){
            src1_slice[j] = src1[j] & mask; //get the slices of all elements in the array
        }

        status = pimCopyHostToDevice(PIM_COPY_V, (void*)src1_slice.data(), src_obj[i]);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            return 1;
        }

        //loop to count the occurance of all the possible number in sliced bit
        for (unsigned j = 0; j < radix; j++){
            unsigned brdcast_value = (j << (i * radix_bits)) & mask;
            status = pimBroadCast(PIM_COPY_V, compare_obj[i], brdcast_value);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return 1;
            }

            status = pimEQ(src_obj[i], compare_obj[i], compare_results_obj[i]);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return 1;
            }

            count_table[j] = pimRedSumRanged(compare_results_obj[i], 0, numElements);
        }

        //Assuming the BitSIMD support 8 bits EQ, so CPU doesn't need to creat slice
        // auto start_cpu = high_resolution_clock::now();
        //host do prefix scan on the counting table
        for (unsigned j = 1; j < radix; j++){
            count_table[j] = count_table[j] + count_table[j-1];
        }

        //host perform reording on temp_array and copy it to src1
        std::vector<unsigned> temp_array(numElements);

        for(int j = (int)(numElements - 1); j >= 0; j--){
            unsigned element_num = (src1[j] & mask) >> (i * radix_bits);
            temp_array[count_table[element_num]-1] = src1[j];
            count_table[element_num]--;
        }
        src1 = temp_array;
        // auto stop_cpu = high_resolution_clock::now();
        // duration_cpu += (stop_cpu - start_cpu);
        //shift mask bit for next iteration
        mask = mask << radix_bits;
    }

    // auto stop_total = high_resolution_clock::now();
    // auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    // auto duration_cpu_total = duration_cast<nanoseconds>(duration_cpu);
    
    // std::cout << "Total execution time = " << duration_total.count() / 1000 << "ms" << std::endl;
    // std::cout << "CPU execution time = " << duration_cpu_total.count() / 1000 << "us" << std::endl;

    // check results
    bool ok = true;
    std::sort(sorted_array.begin(), sorted_array.end());
    if(sorted_array != src1){
        std::cout << "Wrong answer!" << std::endl;
        ok = false;
    }
    pimShowStats();
    if (ok) {
        std::cout << "All correct!" << std::endl;
    }

    return 0;
}