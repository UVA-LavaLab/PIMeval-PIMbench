// Test: C++ version of Advanced Encryption Algorithm (AES)
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include "PIMAuxilary.h"
#include <iostream> 
#include <vector>
#include <inttypes.h>
#include <cstdlib>

#define AES_BLOCK_SIZE 16

// Function-like macros to avoid repetitive code.
#define F(x)   (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1B))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8D : 0))

// Global AES context arrays for key operations.
uint8_t ctx_key[32];
uint8_t ctx_enckey[32];
uint8_t ctx_deckey[32];

// Forward and inverse S-box tables for SubBytes and InvSubBytes steps.
const uint8_t sbox[256] = {
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};
const uint8_t sboxinv[256] = {
      0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

// CPU Function declarations.
uint8_t rjXtime(uint8_t x);
void aesSubBytes(uint8_t *buf);
void aesSubBytesInv(uint8_t *buf);
void aesAddRoundKey(uint8_t *buf, uint8_t *key);
void aesAddRoundKeyCpy(uint8_t *buf, uint8_t *key, uint8_t *cpk);
void aesShiftRows(uint8_t *buf);
void aesShiftRowsInv(uint8_t *buf);
void aesMixColumns(uint8_t *buf);




// Test functions 
int testRjXtime(void);
int testAesSubBytes(void);
int testAesSubBytesInv(void);
int testAesAddRoundKey(void);
int testAesAddRoundKeyCpy(void);
int testAesShiftRows(void);
int testAesShiftRowsInv(void);
int testAesMixColumns(void);

// PIM Function declarations.
PIMAuxilary* rjXtime(PIMAuxilary* x);
void aesSubBytes(std::vector<PIMAuxilary*>* inputObjBuf);
void aesSubBytesInv(std::vector<PIMAuxilary*>* inputObjBuf);
void aesAddRoundKey(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf);
void aesAddRoundKeyCpy(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* cpkObjBuf);
void aesShiftRows(std::vector<PIMAuxilary*>* inputObjBuf);
void aesShiftRowsInv(std::vector<PIMAuxilary*>* inputObjBuf);
void aesMixColumns(std::vector<PIMAuxilary*>* inputObjBuf);


int main(){
    if (int return_status = testAesShiftRowsInv())
    {
        std::cout << "Error in function test_rj_xtime" << std::endl;
        return return_status;
    }
    return 0;
}

uint8_t rjXtime(uint8_t x){
    return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
} 

uint8_t rjXtimeV2(uint8_t x){
    uint8_t shifted = x << 1;
    uint8_t mask = x;
    uint8_t returnValue = 0x1b;
    uint8_t const1 = 0x80;
    mask = mask & const1;
    mask = mask >> 7;  
    returnValue = returnValue * mask;
    returnValue = shifted ^ returnValue;
    // std::cout << "[DEBUG] returnValue: " << (int) returnValue << std::endl; 
    return returnValue;
} 

// subbyte operation
void aesSubBytes(uint8_t *buf){
    uint8_t b;
    for (unsigned j = 0; j < 16; ++j)
    {
        b = buf[j];
        buf[j] = sbox[b];
    }
} 

// inv subbyte operation
void aesSubBytesInv(uint8_t *buf){
    uint8_t b;
    for (unsigned j = 0; j < 16; ++j)
    {
        b = buf[j];
        buf[j] = sboxinv[b];
    }
}


// shift row operation
void aesShiftRows(uint8_t *buf){
    uint8_t i, j;
    i = buf[1];
    buf[1] = buf[5];
    buf[5] = buf[9];
    buf[9] = buf[13];
    buf[13] = i;
    i = buf[10];
    buf[10] = buf[2];
    buf[2] = i;
    j = buf[3];
    buf[3] = buf[15];
    buf[15] = buf[11];
    buf[11] = buf[7];
    buf[7] = j;
    j = buf[14];
    buf[14] = buf[6];
    buf[6]  = j;
}


// shift row operation
void aesShiftRowsInv(uint8_t *buf){
    uint8_t i, j;
    i = buf[1];
    buf[1] = buf[13];
    buf[13] = buf[9];
    buf[9] = buf[5];
    buf[5] = i;
    i = buf[2];
    buf[2] = buf[10];
    buf[10] = i;
    j = buf[3];
    buf[3] = buf[7];
    buf[7] = buf[11];
    buf[11] = buf[15];
    buf[15] = j;
    j = buf[6];
    buf[6] = buf[14];
    buf[14] = j;
}

void aesAddRoundKey(uint8_t *buf, uint8_t *key)
{
    for (int j = 15; j >= 0; j--){
        buf[j] ^= key[j];
    }
}

void aesAddRoundKeyCpy(uint8_t *buf, uint8_t *key, uint8_t *cpk){ 
    uint8_t j = 16;
    while (j--){
        cpk[j] = key[j];
        buf[j] ^= cpk[j];
        cpk[16 + j] = key[16 + j];
    }

}

void aesMixColumns(uint8_t *buf) {
    uint8_t j, a, b, c, d, e, f;
    for (j = 0; j < 16; j += 4){
        a = buf[j];
        b = buf[j + 1];
        c = buf[j + 2];
        d = buf[j + 3];
        e = a ^ b;
        e ^= c;
        e ^= d;

        f = a ^ b;
        f = rjXtime(f);
        f ^= e;
        buf[j] ^= f;

        b ^= c; 
        b = rjXtime(b);
        b ^= e; 
        buf[j+1] ^= b;

        c ^= d; 
        c = rjXtime(c);
        c ^= e; 
        buf[j+2] ^= c;

        
        d ^= a; 
        d = rjXtime(d);
        d ^= e; 
        buf[j+3] ^= d;
    }
}

int testRjXtime(void){

    std::cout << "PIM test: Matrix vector multiplication" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 8;
    unsigned totalElementCount = numCores * numCols;

    
    PIMAuxilary* xObj = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    PIMAuxilary* zObj = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, xObj->pimObjId, PIM_INT32);


    // Initialize x 
    for (unsigned i = 0; i < totalElementCount; ++i) {
        // xObj->array[i] = i % 256;
        xObj->array[i] = 0x55;
        
    }

    // Copy x to the device 
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)xObj->array.data(), xObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    
    zObj = rjXtime(xObj);


    uint8_t x = 0x55;
    uint8_t z2 = rjXtimeV2(x);


    pimShowStats();

    if (zObj->array[0] == z2) {
        std::cout << "zObj->array[0]" << zObj->array[0] << std::endl;
        std::cout << "All correct!" << std::endl;

        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    
    return 0;
}

int testAesSubBytes(void) {
    std::cout << "PIM test: AES.testAesSubBytes" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 32;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    }

    
    
    // Initialize buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }



    
    aesSubBytes(&inputObjBuf);
    aesSubBytes(buf);



    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << inputObjBuf[0]->array[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesSubBytesInv(void) {
    std::cout << "PIM test: AES.testAesSubBytesInv" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 8;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    }

    
    
    // Initialize buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }



    
    aesSubBytesInv(&inputObjBuf);
    aesSubBytesInv(buf);



    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << inputObjBuf[0]->array[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesAddRoundKey(void) {
    std::cout << "PIM test: AES.testAesAddRoundKey" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 8;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    inputObjBuf[0] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    for (unsigned j = 1; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(inputObjBuf[0]);
    }

    std::vector<PIMAuxilary*> keyObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        keyObjBuf[j] = new PIMAuxilary(inputObjBuf[0]);
    }

    
    // Initialize buffer 
    uint8_t buf[16]; 
    uint8_t key[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
        key[j] = 0xaa;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
            keyObjBuf[j]->array[i] = key[j];
        }    
    }
    
    aesAddRoundKey(&inputObjBuf, &keyObjBuf);
    aesAddRoundKey(buf, key);

    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << (int)inputObjBuf[0]->array[0] << std::endl;
    std::cout << "buf[0]: " << (int)buf[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesAddRoundKeyCpy(void) {
    std::cout << "PIM test: AES.testAesAddRoundKeyCpy" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 8;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    inputObjBuf[0] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    for (unsigned j = 1; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(inputObjBuf[0]);
    }

    std::vector<PIMAuxilary*> keyObjBuf(32);
    for (unsigned j = 0; j < 32; ++j) {
        keyObjBuf[j] = new PIMAuxilary(inputObjBuf[0]);
    }

    std::vector<PIMAuxilary*> cpkObjBuf(32);
    for (unsigned j = 0; j < 32; ++j) {
        cpkObjBuf[j] = new PIMAuxilary(inputObjBuf[0]);
    }

    
    // Initialize input buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }

    
    // Initialize key buffers 
    uint8_t key[32]; 
    uint8_t cpk[32]; 
    for (unsigned j = 0; j < 16; ++j) {
        key[j] = 0xaa;
        cpk[j] = 0xaa;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            keyObjBuf[j]->array[i] = key[j];
            cpkObjBuf[j]->array[i] = cpk[j];
        }    
    }


    
    aesAddRoundKeyCpy(&inputObjBuf, &keyObjBuf, &cpkObjBuf);
    aesAddRoundKeyCpy(buf, key, cpk);

    pimShowStats();
    std::cout << "cpkObjBuf[0]->array[0]: " << (int)cpkObjBuf[0]->array[0] << std::endl;
    std::cout << "cpk[0]: " << (int)cpk[0] << std::endl;
    if (cpkObjBuf[0]->array[0] == cpk[0]) {
        
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesShiftRows(void) {
    std::cout << "PIM test: AES.aesShiftRows" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 32;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    }

    
    
    // Initialize buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }



    
    aesShiftRows(&inputObjBuf);
    aesShiftRows(buf);



    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << inputObjBuf[0]->array[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesShiftRowsInv(void) {
    std::cout << "PIM test: AES.aesShiftRowsInv" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 32;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    }

    
    
    // Initialize buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }



    
    aesShiftRowsInv(&inputObjBuf);
    aesShiftRowsInv(buf);



    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << inputObjBuf[0]->array[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

int testAesMixColumns(void) {
    std::cout << "PIM test: AES.aesMixColumns" << std::endl;

    unsigned numCores = 4;
    unsigned numRows = 65536;
    unsigned numCols = 1024;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    unsigned bitsPerElement = 32;
    unsigned totalElementCount = numCores * numCols;

    std::vector<PIMAuxilary*> inputObjBuf(16);
    for (unsigned j = 0; j < 16; ++j) {
        inputObjBuf[j] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
    }

    
    
    // Initialize buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = 0x55;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < totalElementCount; ++i) {
            inputObjBuf[j]->array[i] = buf[j];
        }    
    }



    
    aesMixColumns(&inputObjBuf);
    aesMixColumns(buf);



    pimShowStats();
    std::cout << "inputObjBuf[0]->array[0]: " << inputObjBuf[0]->array[0] << std::endl;
    if (inputObjBuf[0]->array[0] == buf[0]) {
        std::cout << "All correct!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Abort" << std::endl;
        abort();
    }
    return 0;
}

PIMAuxilary* rjXtime(PIMAuxilary* xObj){
    int status; 

    /* TODO: Change with the real PIM API */
    // uint8_t shifted = x << 1;
    PIMAuxilary* shiftedObj = new PIMAuxilary(xObj);
    pimShiftLeft(shiftedObj, 1); 
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)shiftedObj->array.data(), shiftedObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    /* END */

    
    // uint8_t mask = x;
    PIMAuxilary* maskObj = new PIMAuxilary(xObj);
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)xObj->array.data(), maskObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    
    // uint8_t returnValue = 0x1b;
    PIMAuxilary* returnValueObj = new PIMAuxilary(xObj);
    for (unsigned i = 0; i < returnValueObj->numElements; ++i) {
        returnValueObj->array[i] =  0x1b;
    }
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)returnValueObj->array.data(), returnValueObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }

    // uint8_t const1 = 0x80;
    PIMAuxilary* const1Obj = new PIMAuxilary(xObj);
    for (unsigned i = 0; i < const1Obj->numElements; ++i) {
        const1Obj->array[i] =  0x80;
    }
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)const1Obj->array.data(), const1Obj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }


    // mask = mask & const1;
    status = pimAnd(maskObj->pimObjId, const1Obj->pimObjId, maskObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    pimFree(const1Obj->pimObjId);

    

    /* TODO: Change with the real PIM API */
    // mask = mask >> 7;  
    status = pimCopyDeviceToHost(PIM_COPY_V, maskObj->pimObjId, (void*)const1Obj->array.data());
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    pimShiftRight(maskObj, 7); 
    status = pimCopyHostToDevice(PIM_COPY_V, (void*)maskObj->array.data(), maskObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    /* END */


    /* TODO: Change with the real PIM API */
    /* Can be replaced by row replicate function*/
    // returnValue = returnValue * mask;
    status = pimMul(returnValueObj->pimObjId, maskObj->pimObjId, returnValueObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    /* END */
    pimFree(maskObj->pimObjId);

    // returnValue = shifted ^ returnValue;
    status = pimXor(returnValueObj->pimObjId, shiftedObj->pimObjId, returnValueObj->pimObjId);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    status = pimCopyDeviceToHost(PIM_COPY_V, returnValueObj->pimObjId, (void*)returnValueObj->array.data());
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    pimFree(shiftedObj->pimObjId);

    // std::cout << "[DEBUG] returnValueObj: " << (int) returnValueObj->array[0] << std::endl; 
    return returnValueObj;


    
}

void aesSubBytes(std::vector<PIMAuxilary*>* inputObjBuf) {
    // int status;
    
    // Copy input buffer to the device 
    // for (unsigned j = 0; j < 16; ++j) {
    //     status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
    //     if (status != PIM_OK) {
    //         std::cout << "Abort" << std::endl;
    //         abort();
    //     }
    // }

    // /* TODO: Implementation based on bit-serial look-up table */
    // // Copy input buffer back to the host 
    // for (unsigned j = 0; j < 16; ++j) {
    //     status = pimCopyDeviceToHost(PIM_COPY_V,(*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
    //     if (status != PIM_OK) {
    //         std::cout << "Abort" << std::endl;
    //         abort();
    //     }
    // }

    uint8_t b;
    for (unsigned j = 0; j < 16; ++j)
    {
        for (unsigned i = 0; i < (*inputObjBuf)[0]->numElements; ++i) {
            b = (*inputObjBuf)[j]->array[i];
            (*inputObjBuf)[j]->array[i] = sbox[b];
        }
        
    }
    /* END */



}

void aesSubBytesInv(std::vector<PIMAuxilary*>* inputObjBuf) {
    // int status;
    
    // Copy input buffer to the device 
    // for (unsigned j = 0; j < 16; ++j) {
    //     status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
    //     if (status != PIM_OK) {
    //         std::cout << "Abort" << std::endl;
    //         abort();
    //     }
    // }

    // /* TODO: Implementation based on bit-serial look-up table */
    // // Copy input buffer back to the host 
    // for (unsigned j = 0; j < 16; ++j) {
    //     status = pimCopyDeviceToHost(PIM_COPY_V,(*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
    //     if (status != PIM_OK) {
    //         std::cout << "Abort" << std::endl;
    //         abort();
    //     }
    // }

    uint8_t b;
    for (unsigned j = 0; j < 16; ++j)
    {
        for (unsigned i = 0; i < (*inputObjBuf)[0]->numElements; ++i) {
            b = (*inputObjBuf)[j]->array[i];
            (*inputObjBuf)[j]->array[i] = sboxinv[b];
        }
        
    }
    /* END */



}

void aesAddRoundKey(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf) {
    int status;
    
    // Copy input buffer to the device 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }

    // Copy key buffer to the device 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*keyObjBuf)[j]->array.data(), (*keyObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }


    /* for (int j = 0; j < 16; j++){ 
        buf[j] ^= key[j];
    } */
    for (int j = 0; j < 16; j++){ 
        status = pimXor((*inputObjBuf)[j]->pimObjId, (*keyObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }   



    // Copy input buffer back to the host 
    for (unsigned j = 0; j < 16; ++j) {
        std::cout << j << std::endl;
        status = pimCopyDeviceToHost(PIM_COPY_V, (*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
        if (status != PIM_OK) {

            std::cout << "Abort" << std::endl;
            abort();
        }
    }
}

void aesAddRoundKeyCpy(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* cpkObjBuf) {
int status;
    
    // Copy input buffer to the device 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }

    // Copy key buffer to the device 
    for (unsigned j = 0; j < 32; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*keyObjBuf)[j]->array.data(), (*keyObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }


    /* TODO: Copy from keyObjBuf in the device instead of host*/
    /* for (unsigned j = 0; j < 16; ++j) {
        cpk[i] = key[i] 
    } */
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*keyObjBuf)[j]->array.data(), (*cpkObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }
    /* END */


    /* for (int j = 0; j < 16; j++){ 
       buf[j] ^= cpk[j];
    } */
    for (int j = 0; j < 16; j++){ 
        status = pimXor((*inputObjBuf)[j]->pimObjId, (*cpkObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }   

    /* TODO: Copy from keyObjBuf in the device instead of host*/
    /* for (int j = 0; j < 16; j++){ 
       cpk[16 + j] = key[16 + j];
    } */
    for (int j = 0; j < 16; j++){ 
        status = pimCopyDeviceToHost(PIM_COPY_V, (*keyObjBuf)[16 + j]->pimObjId, (void*)(*keyObjBuf)[16 + j]->array.data());
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        status = pimCopyHostToDevice(PIM_COPY_V, (*keyObjBuf)[16 + j]->array.data(), (*cpkObjBuf)[16 + j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

    } 
    /* TODO: Copy from keyObjBuf in the device instead of host*/  



    // Copy input buffer back to the host 
    for (unsigned j = 0; j < 16; ++j) {
        std::cout << j << std::endl;
        status = pimCopyDeviceToHost(PIM_COPY_V, (*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
        if (status != PIM_OK) {

            std::cout << "Abort" << std::endl;
            abort();
        }
    }

    // Copy cpk buffer back to the host 
    for (unsigned j = 0; j < 32; ++j) {
        std::cout << j << std::endl;
        status = pimCopyDeviceToHost(PIM_COPY_V, (*keyObjBuf)[j]->pimObjId, (void*)(*keyObjBuf)[j]->array.data());
        if (status != PIM_OK) {

            std::cout << "Abort" << std::endl;
            abort();
        }
    }
    
}

void aesShiftRows(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status;
    int i, j;

    // Copy input buffer to the device 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }

    i = (*inputObjBuf)[1]->pimObjId;
    (*inputObjBuf)[1]->pimObjId = (*inputObjBuf)[5]->pimObjId;
    (*inputObjBuf)[5]->pimObjId = (*inputObjBuf)[9]->pimObjId;
    (*inputObjBuf)[9]->pimObjId = (*inputObjBuf)[13]->pimObjId;
    (*inputObjBuf)[13]->pimObjId = i; 
    i = (*inputObjBuf)[10]->pimObjId;
    (*inputObjBuf)[10]->pimObjId = (*inputObjBuf)[2]->pimObjId;
    (*inputObjBuf)[2]->pimObjId = i;
    j = (*inputObjBuf)[3]->pimObjId;
    (*inputObjBuf)[3]->pimObjId = (*inputObjBuf)[15]->pimObjId;
    (*inputObjBuf)[15]->pimObjId = (*inputObjBuf)[11]->pimObjId;
    (*inputObjBuf)[11]->pimObjId = (*inputObjBuf)[7]->pimObjId;
    (*inputObjBuf)[7]->pimObjId = j;
    j = (*inputObjBuf)[14]->pimObjId;
    (*inputObjBuf)[14]->pimObjId = (*inputObjBuf)[6]->pimObjId;
    (*inputObjBuf)[6]->pimObjId = j;

    // Copy input buffer back to the host 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyDeviceToHost(PIM_COPY_V,(*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }
}

void aesShiftRowsInv(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status;
    int i, j;

    // Copy input buffer to the device 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }

    i = (*inputObjBuf)[1]->pimObjId;
    (*inputObjBuf)[1]->pimObjId = (*inputObjBuf)[13]->pimObjId;
    (*inputObjBuf)[13]->pimObjId = (*inputObjBuf)[9]->pimObjId;
    (*inputObjBuf)[9]->pimObjId = (*inputObjBuf)[5]->pimObjId;
    (*inputObjBuf)[5]->pimObjId = i; 
    i = (*inputObjBuf)[2]->pimObjId;
    (*inputObjBuf)[2]->pimObjId = (*inputObjBuf)[10]->pimObjId;
    (*inputObjBuf)[10]->pimObjId = i;
    j = (*inputObjBuf)[3]->pimObjId;
    (*inputObjBuf)[3]->pimObjId = (*inputObjBuf)[7]->pimObjId;
    (*inputObjBuf)[7]->pimObjId = (*inputObjBuf)[11]->pimObjId;
    (*inputObjBuf)[11]->pimObjId = (*inputObjBuf)[15]->pimObjId;
    (*inputObjBuf)[15]->pimObjId = j;
    j = (*inputObjBuf)[6]->pimObjId;
    (*inputObjBuf)[6]->pimObjId = (*inputObjBuf)[14]->pimObjId;
    (*inputObjBuf)[14]->pimObjId = j;

    // Copy input buffer back to the host 
    for (unsigned j = 0; j < 16; ++j) {
        status = pimCopyDeviceToHost(PIM_COPY_V,(*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
    }
}

void aesMixColumns(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status; 
    

    // uint8_t a, b, c, d, e, f;
    PIMAuxilary* aObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* bObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* cObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* dObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* eObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* fObj = new PIMAuxilary((*inputObjBuf)[0]);


    for (int j = 0; j < 16; j += 4){
        //  a = buf[j];
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j]->array.data(), aObj->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // b = buf[j + 1];
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j + 1]->array.data(), bObj->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // c = buf[j + 2];
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j + 2]->array.data(), cObj->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // d = buf[j + 2];
        status = pimCopyHostToDevice(PIM_COPY_V, (void*)(*inputObjBuf)[j + 3]->array.data(), dObj->pimObjId);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // e = a ^ b;
        status = pimXor(aObj->pimObjId, dObj->pimObjId, eObj->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // e ^= c;
        status = pimXor(eObj->pimObjId, cObj->pimObjId, eObj->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }
        
        // e ^= d;
        status = pimXor(eObj->pimObjId, dObj->pimObjId, eObj->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // f = a ^ b;
        status = pimXor(aObj->pimObjId, bObj->pimObjId, fObj->pimObjId); 
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            abort();
        }

        // f = rjXtime(f);
        /* TODO finish this function*/


    }

}       