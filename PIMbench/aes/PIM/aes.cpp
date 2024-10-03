// Test: C++ version of Advanced Encryption Algorithm (AES)
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "PIMAuxilary.h"
#include <iostream> 
#include <vector>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <getopt.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define AES_BLOCK_SIZE 16

#define SUBBYTES_FUNCTIONAL
#define RJXTIME_FUCNTIONAL

// Function-like macros to avoid repetitive code.
#define F(x)   (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1B))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8D : 0))

// Global AES context arrays for key operations
uint8_t ctx_key_global[32] = {0};
uint8_t ctx_enckey_global[32] = {0};
uint8_t ctx_deckey_global[32] = {0};
std::chrono::duration<double, std::milli> host_elapsedTime_global;

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

// Params 
typedef struct Params
{
    uint64_t inputSize;
    const char *keyFile;
    const char *inputFile;
    const char *cipherFile;
    const char *outputFile;
    bool shouldVerify;
} Params;
 
// Util functions 
void inline interrupt(int line); 
void bufferPadding(uint8_t* inBuf, uint8_t* outBuf, unsigned inNumBytes, unsigned outNumBytes);
int compare_files(const char *file1, const char *file2);
void usage();
struct Params getInputParams(int argc, char **argv); 

  
// CPU Function declarations.
uint8_t rjXtime(uint8_t x);
void aesSubBytes(uint8_t *buf);
void aesSubBytesInv(uint8_t *buf);
void aesAddRoundKey(uint8_t *buf, uint8_t *key);
void aesAddRoundKeyCpy(uint8_t *buf, uint8_t *key, uint8_t *cpk);
void aesShiftRows(uint8_t *buf);
void aesShiftRowsInv(uint8_t *buf);
void aesMixColumns(uint8_t *buf);
void aesMixColumnsInv(uint8_t *buf);
void aesExpandEncKey(uint8_t *k, uint8_t *rc, const uint8_t *sb);
void aesExpandDecKey(uint8_t *k, uint8_t *rc);
void aes256Init(uint8_t *k);
void aes256EncryptEcb(uint8_t *buf, unsigned long offset);
void aes256DecryptEcb(uint8_t *buf, unsigned long offset);
void encryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes);
void decryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes);

#define FUNCTION_UNDER_TEST testDemo  

// Test functions 
int testRjXtime(void);  
int testAesSubBytes(void);
int testAesSubBytesInv(void);
int testAesAddRoundKey(void);
int testAesAddRoundKeyCpy(void);
int testAesShiftRows(void);
int testAesShiftRowsInv(void);
int testAesMixColumns(void); 
int testAesMixColumnsInv(void); 
int testAes256EncryptEcb(void); 
int testAes256DecryptEcb(void);
int testEncryptdemo(void); 
int testDecryptdemo(void);
int testDemo(int argc, char **argv);

// PIM Function declarations.
void rjXtime(PIMAuxilary* xObj, PIMAuxilary* returnValueObj);
void aesSubBytes(std::vector<PIMAuxilary*>* inputObjBuf);
void aesSubBytesInv(std::vector<PIMAuxilary*>* inputObjBuf);
void aesAddRoundKey(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* outObjBuf) ;
void aesAddRoundKeyCpy(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* cpkObjBuf, std::vector<PIMAuxilary*>* outObjBuf);

void aesShiftRows(std::vector<PIMAuxilary*>* inputObjBuf);
void aesShiftRowsInv(std::vector<PIMAuxilary*>* inputObjBuf);
void aesMixColumns(std::vector<PIMAuxilary*>* inputObjBuf);
void aesMixColumnsInv(std::vector<PIMAuxilary*>* inputObjBuf);
void aes256EncryptEcb(std::vector<PIMAuxilary*>* inputObjBuf, unsigned long offset);
void aes256DecryptEcb(std::vector<PIMAuxilary*>* inputObjBuf, unsigned long offset);
void encryptdemo(uint8_t key[32], std::vector<PIMAuxilary*>* inputObjBuf, unsigned long numCalls);
void decryptdemo(uint8_t key[32], std::vector<PIMAuxilary*>* inputObjBuf, unsigned long numCalls);
   
int main(int argc, char **argv){
    srand(time(NULL));
    // int returnStatus = testEncryptdemo();
    // int returnStatus = testDecryptdemo();
    // int returnStatus = FUNCTION_UNDER_TEST();
    // std::cout << "INFO: Host elapsed time : " << std::fixed << std::setprecision(6) << host_elapsedTime_global.count() << " ms." << std::endl;
   
    int returnStatus = testDemo(argc, argv);
    /* TODO: uncomment it. used to avoid core dump in slurm task. */
    assert(returnStatus == 0);
    return 0;
}

uint8_t rjXtime(uint8_t x){
    uint8_t shifted = x << 1;
    uint8_t mask = x;
    uint8_t returnValue = 0x1b;
    uint8_t const1 = 0x80;
    mask = mask & const1;
    mask = mask >> 7;  
    const1 = returnValue * mask;
    returnValue = shifted ^ const1;

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
    for (int j = AES_BLOCK_SIZE - 1; j >= 0; j--){
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
    uint8_t j, a, b, c, d, e, f, t;
    for (j = 0; j < 16; j += 4){
        a = buf[j];
        b = buf[j + 1];
        c = buf[j + 2];
        d = buf[j + 3];

        e = a ^ b;
        t = e ^ c; 
        e = t ^ d;

        t = a ^ b;  
        f = rjXtime(t);
        t = f ^ e;
        f = buf[j] ^ t;
        buf[j] = f;

        t = b ^ c; 
        b = rjXtime(t);

        t = b ^ e; 
        b = buf[j+1] ^ t;    
        buf[j+1] = b;

        t = c ^ d;
        c = rjXtime(t);
        t = c ^ e; 
        c = buf[j+2] ^ t;
        buf[j+2] = c;

        t = d ^ a;
        d = rjXtime(t);
        t = d ^ e; 
        d = buf[j+3] ^ t;
        buf[j+3] = d;    
    }
}

void aesMixColumnsInv(uint8_t *buf) {
    uint8_t j, a, b, c, d, e, x, y, z;
    uint8_t t0, t;
    for (j = 0; j < 16; j += 4){
        a = buf[j];
        b = buf[j + 1];
        c = buf[j + 2];
        d = buf[j + 3];

        e = a ^ b;
        t = e ^ c;
        e = t ^ d;
        z = rjXtime(e);

        t = a ^ c;
        t0 = t ^ z;
        t = rjXtime(t0);
        t0 = rjXtime(t);
        x = e ^ t0;

        t = b ^ d;
        t0 = t ^ z;
        t = rjXtime(t0);
        t0 = rjXtime(t);
        y = e ^ t0;

        t0 = a ^ b;
        t0 = rjXtime(t0);
        t = t0 ^ x;
        t0 = buf[j] ^ t;
        buf[j] = t0;

        t0 = b ^ c;
        t = rjXtime(t0);
        t0 = t ^ y;
        t = buf[j + 1] ^ t0;
        buf[j + 1] = t;

        t0 = c ^ d;
        t = rjXtime(t0);
        t0 = t ^ x;
        t = buf[j + 2] ^ t0;
        buf[j + 2] = t;

        t0 = d ^ a;
        t = rjXtime(t0);
        t0 = t ^ y;
        t = buf[j + 3] ^ t0;
        buf[j + 3] = t;
    }
}

// aes encrypt algorithm
void aes256EncryptEcb(uint8_t *buf, unsigned long offset){
    uint8_t l = 1, rcon;
    uint8_t bufT[AES_BLOCK_SIZE];
    uint8_t ctx_enckey_local[32];
    uint8_t ctx_key_local[32];
    memcpy(bufT, &buf[offset], AES_BLOCK_SIZE);
    memcpy(ctx_enckey_local, ctx_enckey_global, 32);
    memcpy(ctx_key_local, ctx_key_global, 32);

    aesAddRoundKeyCpy(bufT, ctx_enckey_local, ctx_key_local);

    for(l = 1, rcon = l; l < 14; ++l){
        aesSubBytes(bufT);
        aesShiftRows(bufT);   
        aesMixColumns(bufT);
        if( l & 1 ){
            aesAddRoundKey(bufT, ctx_key_local);
        }
        else{
            aesExpandEncKey(ctx_key_local, &rcon, sbox);
            aesAddRoundKey(bufT, ctx_key_local);
        }
    }
    aesSubBytes(bufT);
    aesShiftRows(bufT);
    aesExpandEncKey(ctx_key_local, &rcon, sbox);
    aesAddRoundKey(bufT, ctx_key_local);
    memcpy(&buf[offset], bufT, AES_BLOCK_SIZE);
}

// aes decrypt algorithm
void aes256DecryptEcb(uint8_t *buf, unsigned long offset){
    uint8_t l, rcon;
    uint8_t bufT[AES_BLOCK_SIZE];
    uint8_t ctx_deckey_local[32];
    uint8_t ctx_key_local[32];

    memcpy(bufT, &buf[offset], AES_BLOCK_SIZE);
    memcpy(ctx_deckey_local, ctx_deckey_global, 32);
    memcpy(ctx_key_local, ctx_key_global, 32);

   

    aesAddRoundKeyCpy(bufT, ctx_deckey_local, ctx_key_local);

    aesShiftRowsInv(bufT);
    aesSubBytesInv(bufT);

    for (l = 14, rcon = 0x80; --l;){
        if((l & 1)){
            
            aesExpandDecKey(ctx_key_local, &rcon);
            aesAddRoundKey(bufT, ctx_key_local);
        }
        else{
            aesAddRoundKey(bufT, ctx_key_local);
        }
        aesMixColumnsInv(bufT);
        aesShiftRowsInv(bufT);
        aesSubBytesInv(bufT);

    }
    aesAddRoundKey(bufT, ctx_key_local);
    memcpy(&buf[offset], bufT, AES_BLOCK_SIZE);
} 

void encryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes){
  aes256Init(key);
  unsigned long offset;

  for (offset = 0; offset < numbytes; offset += AES_BLOCK_SIZE)
    aes256EncryptEcb(buf, offset);
}

// aes decrypt demo
void decryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes){
    aes256Init(key);
    unsigned long offset;
        
    for (offset = 0; offset < numbytes; offset += AES_BLOCK_SIZE)
        aes256DecryptEcb(buf, offset);
}

int testRjXtime(void){

    std::cout << "INFO: PIM test: AES.rjXtime" << std::endl;

    // Configuration parameters
    unsigned numRows = 8192;
    unsigned numRanks = 1; 
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 1024;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numCols = 1024;
    unsigned totalElementCount = numCores * numCols;


    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);

    assert(status == PIM_OK);


    
    PIMAuxilary* xObj = new PIMAuxilary(PIM_ALLOC_AUTO, totalElementCount, PIM_UINT8);
    PIMAuxilary* zObj = new PIMAuxilary(xObj);


    // Initialize x 
    for (unsigned i = 0; i < totalElementCount; ++i) {
        xObj->array[i] = rand() % 256;
    }
    
    // Copy x to the device 
    status = pimCopyHostToDevice((void*)xObj->array.data(), xObj->pimObjId);
    assert(status == PIM_OK);

    rjXtime(xObj, zObj);
    
    pimShowStats();

    status = pimCopyDeviceToHost(zObj->pimObjId, (void*)zObj->array.data());
    assert(status == PIM_OK);

    for (unsigned i = 0; i < totalElementCount; ++i) {
        uint8_t x = xObj->array[i];
        uint8_t z = rjXtime(x);
        if (zObj->array[i] % 256 != z) {
            std::cout << "x: " << (int)x << std::endl;
            std::cout << "zObj->array[" << i << "]: " << (int)zObj->array[i] % 256 << std::endl;
            std::cout << "z: " << (int)z << std::endl;
            return 1; 
        }
    
    }
    
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}
int testAesSubBytes() {
    std::cout << "INFO: PIM test: AES.aesSubBytesInv" << std::endl;

    // Configuration parameters
    unsigned numRanks = 1;
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 1024; 
    unsigned totalCols = numCols * numCores;
    unsigned long numElements = totalCols;
    unsigned long numBytes = numElements * AES_BLOCK_SIZE;
   
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    // Initialize buffer with random values
    uint8_t bufIn[numBytes];
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }

    // Copy data to input object buffer
    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] = bufIn[j];
    }

    // Copy data from host to device
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }


    // Perform AES SubBytes transformation
    for (unsigned int offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE)
        aesSubBytes((uint8_t*)(bufIn + offset));

    aesSubBytes(inputObjBuf);

    // Copy data from device to host
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    pimShowStats();
    // Validate the results
    for (unsigned j = 0; j < numBytes; ++j) {
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] != bufIn[j]) {
            for (unsigned i = j; i < j + 1; ++i) {
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[" << i / AES_BLOCK_SIZE << "]: " 
                          << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] << std::endl;
                std::cout << "bufIn[" << i << "]: " << (int)bufIn[i] << std::endl;
            }
            return 1;
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAesSubBytesInv() {
    std::cout << "INFO: PIM test: AES.aesSubBytesInv" << std::endl;

    // Configuration parameters
    unsigned numRanks = 1;
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 1024; 
    unsigned totalCols = numCols * numCores;
    unsigned long numElements = totalCols;
    unsigned long numBytes = numElements * AES_BLOCK_SIZE;
   
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    // Initialize buffer with random values
    uint8_t bufIn[numBytes];
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }

    // Copy data to input object buffer
    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] = bufIn[j];
    }

    // Copy data from host to device
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    // Copy data from device to host (initial validation)
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());

        assert(status == PIM_OK);
    }

    // Perform AES SubBytesInv transformation
    for (unsigned int offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE)
        aesSubBytesInv((uint8_t*)(bufIn + offset));
    aesSubBytesInv(inputObjBuf);

    // Copy data from device to host (after transformation)
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    pimShowStats();
    // Validate the results
    for (unsigned j = 0; j < numBytes; ++j) {
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] != bufIn[j]) {
            for (unsigned i = j; i < j + 1; ++i) {
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[" << i / AES_BLOCK_SIZE << "]: " 
                          << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] << std::endl;
                std::cout << "bufIn[" << i << "]: " << (int)bufIn[i] << std::endl;
            }
            return 1;
        }
    }


    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAesAddRoundKey() {
    std::cout << "INFO: PIM test: AES.testAesAddRoundKey" << std::endl;

    // Configuration parameters
    unsigned numRanks = 1;
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 1024; 
    unsigned totalCols = numCols * numCores;
    unsigned long numElements = totalCols;
    unsigned long numBytes = numElements * AES_BLOCK_SIZE;
   
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }
    
    std::vector<PIMAuxilary*> *keyObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*keyObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*> *outObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*outObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    // Initialize buffers with values
    uint8_t buf[AES_BLOCK_SIZE];
    uint8_t key[AES_BLOCK_SIZE];
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        buf[j] = rand() % 256; 
        key[j] = rand() % 256; // Random value for key
    }

    // Copy data to input and key object buffers
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        for (unsigned i = 0; i < numElements; ++i) {
            (*inputObjBuf)[j]->array[i] = buf[j];
            (*keyObjBuf)[j]->array[i] = key[j];
            (*outObjBuf)[j]->array[i] = 0;
            
        }    
    }

    // Copy data from host to device
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*keyObjBuf)[j]->array.data(), (*keyObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    // Perform AES AddRoundKey transformation
    aesAddRoundKey(inputObjBuf, keyObjBuf, outObjBuf);
    aesAddRoundKey(buf, key);
    
    // Copy data from device to host
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {

        status = pimCopyDeviceToHost((*outObjBuf)[j]->pimObjId, (*outObjBuf)[j]->array.data());

        assert(status == PIM_OK);
    }
   
    // Validate the results
    pimShowStats();
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) { 
        if ((*outObjBuf)[j]->array[0] != buf[j]) {
            return 1;
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAesAddRoundKeyCpy(void) {
    std::cout << "INFO: PIM test: AES.testAesAddRoundKeyCpy" << std::endl;

    // Configuration parameters
    unsigned numRanks = 1;
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 1024; 
    unsigned totalCols = numCols * numCores;
    unsigned long numElements = totalCols;
    unsigned long numBytes = numElements * AES_BLOCK_SIZE;
   
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*> *keyObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*keyObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*> *cpkObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*cpkObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*> *outputObjBuf = new std::vector<PIMAuxilary*>(16);
    for (unsigned j = 0; j < 16; ++j) {
        (*outputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    // Initialize input buffer 
    uint8_t buf[16]; 
    for (unsigned j = 0; j < 16; ++j) {
        buf[j] = rand() % 256;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < numElements; ++i) {
            (*inputObjBuf)[j]->array[i] = buf[j];
        }    
    }
     
    // Initialize key buffers 
    uint8_t key[32]; 
    uint8_t cpk[32]; 
    for (unsigned j = 0; j < 32; ++j) {
        key[j] = rand() % 256;
        cpk[j] = rand() % 256;
    }

    for (unsigned j = 0; j < 16; ++j) {
        for (unsigned i = 0; i < numElements; ++i) {
            (*keyObjBuf)[j]->array[i] = key[j];
        }    
    }
    
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    

    for (unsigned j = 0; j < 32; ++j) {
        status = pimCopyHostToDevice((*keyObjBuf)[j]->array.data(), (*keyObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }


    aesAddRoundKeyCpy(inputObjBuf, keyObjBuf, cpkObjBuf, outputObjBuf);
    aesAddRoundKeyCpy(buf, key, cpk);

    
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*outputObjBuf)[j]->pimObjId, (*outputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    for (unsigned j = 0; j < 32; ++j) {
        status = pimCopyDeviceToHost((*cpkObjBuf)[j]->pimObjId, (*cpkObjBuf)[j]->array.data());

        assert(status == PIM_OK);
    }

    pimShowStats();
    
    for (int j = 0; j < AES_BLOCK_SIZE; ++j)
    {
        if ((*cpkObjBuf)[j]->array[0] != cpk[j]) {
                std::cout << "cpkObjBuf[" << j << "]->array[0]: " << (int)(*cpkObjBuf)[j]->array[0] << std::endl;
                std::cout << "cpk[" << j <<"]: " << (int)cpk[j] << std::endl;
                std::cout << "Abort" << std::endl;
                return 1;
                
        }
        if ((*outputObjBuf)[j]->array[0] != buf[j]) {
                std::cout << "outputObjBuf[" << j << "]->array[0]: " << (int)(*outputObjBuf)[j]->array[0] << std::endl;
                std::cout << "buf[" << j <<"]: " << (int)buf[j] << std::endl;
                std::cout << "Abort" << std::endl;
                return 1;
                
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAesShiftRows(void) {
    std::cout << "INFO: PIM test: AES.aesShiftRows" << std::endl;

    // Configuration parameters
    unsigned numRanks = 1;
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numRanks * numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 1024; 
    unsigned totalCols = numCols * numCores;
    unsigned long numElements = totalCols;
    unsigned long numBytes = numElements * AES_BLOCK_SIZE;
   
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBanks, numSubarrayPerBank, numRows, numCols);


    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);

    }

    // Initialize input buffer with random values
    uint8_t bufIn[numBytes];
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = j % AES_BLOCK_SIZE; // rand() % 256;
    }

    // Copy input buffer values to inputObjBuf
    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] = bufIn[j];
    }

    // Copy data from host to device
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

   
    // Perform AES ShiftRows operation on the buffer
    for (unsigned offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        aesShiftRows(bufIn + offset);
    }
    aesShiftRows(inputObjBuf);

    // Copy data back from device to host after AES ShiftRows operation
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    // Show performance statistics
    pimShowStats();

    // Verify the results
    for (unsigned j = 0; j < numBytes; ++j) {
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] != bufIn[j]) {
            // Output the details of mismatch
            for (unsigned i = j; i < j + 1; ++i) {
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[" << i / AES_BLOCK_SIZE << "]: " 
                          << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] << std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; // Return with error if mismatch is found
        }
    }

    std::cout << "INFO: All correct!" << std::endl;
    return 0; // Return success if all results are correct
}

int testAesShiftRowsInv(void) {
    std::cout << "INFO: PIM test: AES.aesShiftRowsInv" << std::endl;

    // Configuration parameters
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 1024;
    unsigned numCores = numBanks * numSubarrayPerBank;
    unsigned numRows = 65536;

    unsigned numCols = 1024;
    unsigned long numBytes = numCols * AES_BLOCK_SIZE;
    unsigned totalElementCount = numCores * numCols / 2;

    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);

    (*inputObjBuf)[0] = new PIMAuxilary(PIM_ALLOC_V1, totalElementCount, PIM_UINT8);
    for (unsigned j = 1; j < AES_BLOCK_SIZE; ++j) {
        (*inputObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);

    }

    // Initialize input buffer with random values
    uint8_t bufIn[numBytes];
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }

    // Copy input buffer values to inputObjBuf
    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] = bufIn[j];
    }

    // Copy data from host to device
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    // Copy data back from device to host to ensure correctness
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    // Perform AES ShiftRows operation on the buffer
    for (unsigned offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        aesShiftRowsInv(bufIn + offset);
    }
    aesShiftRowsInv(inputObjBuf);

    // Copy data back from device to host after AES ShiftRows operation
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    // Show performance statistics
    pimShowStats();

    // Verify the results
    for (unsigned j = 0; j < numBytes; ++j) {
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE] != bufIn[j]) {
            // Output the details of mismatch
            for (unsigned i = 0; i < numBytes; ++i) {
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[" << i / AES_BLOCK_SIZE << "]: " 
                          << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] << std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; // Return with error if mismatch is found
        }
    }

    std::cout << "INFO: All correct!" << std::endl;
    return 0; // Return success if all results are correct
}

int testAesMixColumns(void) {
    std::cout << "INFO: PIM test: AES.aesMixColumns" << std::endl;

    // Configuration parameters
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 8;
    unsigned numCores = numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 8; 
    unsigned totalElementCount = numCores * numCols;
    unsigned long numBytes = totalElementCount * AES_BLOCK_SIZE;

    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);


    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, totalElementCount, PIM_UINT8);

    for (unsigned j = 1; j < (AES_BLOCK_SIZE); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }

    

    // Initialize buffer 
    uint8_t bufIn[numBytes]; 
   
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256; 
    }


    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % (AES_BLOCK_SIZE)]->array[j / (AES_BLOCK_SIZE)] = bufIn[j];
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    for (int j = 0; j < AES_BLOCK_SIZE; j++)
    {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    for (unsigned offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        aesMixColumns(bufIn + offset);
    }
    aesMixColumns(inputObjBuf);

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    pimShowStats();
    for (unsigned j = 0; j < numBytes; ++j) { 
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE ] % 256 != bufIn[j]) {
        
            for (unsigned i = 0; i < numBytes; ++i) {  
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[ " << i / AES_BLOCK_SIZE << " ]: " << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] % 256<< std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] % 256 << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAesMixColumnsInv(void) {
    std::cout << "INFO: PIM test: AES.testAesMixColumnsInv" << std::endl;

    // Configuration parameters
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 8;
    unsigned numCores = numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 8; 
    unsigned totalElementCount = numCores * numCols;
    unsigned long numBytes = totalElementCount * AES_BLOCK_SIZE;

    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);


    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, totalElementCount, PIM_UINT8);

    for (unsigned j = 1; j < (AES_BLOCK_SIZE); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    

    // Initialize buffer 
    uint8_t bufIn[numBytes]; 
   
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }


    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % (AES_BLOCK_SIZE)]->array[j / (AES_BLOCK_SIZE)] = bufIn[j];
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    for (int j = 0; j < AES_BLOCK_SIZE; j++)
    {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    for (unsigned offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        aesMixColumnsInv(bufIn + offset);
    }
    aesMixColumnsInv(inputObjBuf);

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }

    pimShowStats();
    for (unsigned j = 0; j < numBytes; ++j) { 
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE ] % 256 != bufIn[j]) {
        
            for (unsigned i = 0; i < numBytes; ++i) {  
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[ " << i / AES_BLOCK_SIZE << " ]: " << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] % 256<< std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] % 256 << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testAes256EncryptEcb(void) {
    std::cout << "INFO: PIM test: AES.aes256EncryptEcb" << std::endl;

    // Configuration parameters
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 2;
    unsigned numCores = numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 8; 
    unsigned totalElementCount = numCores * numCols;
    unsigned long numBytes = totalElementCount * AES_BLOCK_SIZE;
    
    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);


    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, totalElementCount, PIM_UINT8);

    for (unsigned j = 1; j < (AES_BLOCK_SIZE); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    

    // Initialize buffer 
    uint8_t bufIn[numBytes]; 
   
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }

    uint8_t key[32]; 
    for (unsigned j = 0; j < 32; ++j) {
        key[j] = rand() % 256; 
    }


    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % (AES_BLOCK_SIZE)]->array[j / (AES_BLOCK_SIZE)] = bufIn[j];
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    for (int offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        memcpy(ctx_key_global, key, 32);
        aes256Init(key);
        aes256EncryptEcb(bufIn, offset);
    }
    
    memcpy(ctx_key_global, key, 32);
    aes256Init(key);

    unsigned long offset = 0;
    aes256EncryptEcb(inputObjBuf, offset);

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }


    pimShowStats();
    for (unsigned j = 0; j < numBytes; ++j) { 
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE ] % 256 != bufIn[j]) {
        
            for (unsigned i = j; i < j + 1; ++i) {  
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[ " << i / AES_BLOCK_SIZE << " ]: " << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] % 256<< std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] % 256 << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }

    std::cout << "INFO: All correct!" << std::endl;
    return 0;

}

int testAes256DecryptEcb(void) {
    std::cout << "INFO: PIM test: AES.aes256DecryptEcb" << std::endl;

    // Configuration parameters
    unsigned numBanks = 1; 
    unsigned numSubarrayPerBank = 8;
    unsigned numCores = numBanks * numSubarrayPerBank / 2;
    unsigned numRows = 65536;

    unsigned numCols = 8; 
    unsigned totalElementCount = numCores * numCols;
    unsigned long numBytes = totalElementCount * AES_BLOCK_SIZE;

    // Initialize PIM device
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, numBanks, numSubarrayPerBank, numRows, numCols);

    // Allocate memory for input buffers
    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);


    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, totalElementCount, PIM_UINT8);

    for (unsigned j = 1; j < (AES_BLOCK_SIZE); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    

    // Initialize buffer 
    uint8_t bufIn[numBytes]; 
   
    for (unsigned j = 0; j < numBytes; ++j) {
        bufIn[j] = rand() % 256;
    }

    uint8_t key[32]; 
    for (unsigned j = 0; j < 32; ++j) {
        key[j] = rand() % 256; 
    }



    for (unsigned j = 0; j < numBytes; ++j) {
        (*inputObjBuf)[j % (AES_BLOCK_SIZE)]->array[j / (AES_BLOCK_SIZE)] = bufIn[j];
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    for (int offset = 0; offset < numBytes; offset += AES_BLOCK_SIZE) {
        memcpy(ctx_key_global, key, 32);
        aes256Init(key);
        aes256EncryptEcb(bufIn, offset);
    }
    
    memcpy(ctx_key_global, key, 32);
    aes256Init(key);

    unsigned long offset = 0;
    aes256EncryptEcb(inputObjBuf, offset);

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }


    pimShowStats();
    for (unsigned j = 0; j < numBytes; ++j) { 
        if ((*inputObjBuf)[j % AES_BLOCK_SIZE]->array[j / AES_BLOCK_SIZE ] != bufIn[j]) {
           
            for (unsigned i = 0; i < numBytes; ++i) {  
                std::cout << "(int)(*inputObjBuf)[" << i % AES_BLOCK_SIZE << "]->array[ " << i / AES_BLOCK_SIZE << " ]: " << (int)(*inputObjBuf)[i % AES_BLOCK_SIZE]->array[i / AES_BLOCK_SIZE] << std::endl;
                std::cout << "buf[" << i << "]: " << (int)bufIn[i] << std::endl;
            }
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;

}

int testEncryptdemo(void) {
    std::cout << "INFO: PIM test: AES.encryptdemo" << std::endl;

    // unsigned long numBytes = 1310720; 
    unsigned long numBytes = 1UL * 1024 * 1024; // * 1024; // 1 GB

    // Each rank has 8 chips; Total Bank = 16; Each Bank contains 32 subarrays;
    unsigned numRanks = 2;
    unsigned numBankPerRank = 1; // 128; // 8 chips * 16 banks
    unsigned numSubarrayPerBank = 4; // 32;
    unsigned numRows = 8192;
    unsigned numCols = 8192;
    unsigned numCores = numRanks * numBankPerRank * numSubarrayPerBank / 2; 
    unsigned totalCols = numCores * numCols;

    unsigned numCalls = 1;
    unsigned numPaddedBufBytes = numBytes;
    unsigned numElements = numPaddedBufBytes / AES_BLOCK_SIZE;

    {
        std::cout << "INFO: numCalls = " << numCalls << std::endl; 
        std::cout << "INFO: numPaddedBufBytes = " << numPaddedBufBytes << std::endl; 
        std::cout << std::endl;

    }
    
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
    assert(status == PIM_OK);

    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE * numCalls);
    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
   
    for (unsigned j = 1; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    
    uint8_t key[32];
    for (unsigned j = 0; j < 32; ++j)
    {
        key[j] = rand() % 256;
    }

    // Allocate the input buffer 
    uint8_t* bufIn = (uint8_t*) malloc(numPaddedBufBytes * sizeof(uint8_t)); 

    // Initialize buffer s
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) {
        bufIn[j] = rand() % 256; 
    }

    // Initialize inputObjBuf
    unsigned words_per_call, i_call, i_chunk, i_aes_block, remained; 
    words_per_call = (totalCols * AES_BLOCK_SIZE);
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) {
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        (*inputObjBuf)[i_aes_block]->array[i_chunk] = bufIn[j]; 
    }

    // Copy inputObjBuf to the device 
    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    memcpy(ctx_key_global, key, 32);
    encryptdemo(key, bufIn, numPaddedBufBytes);   

    memcpy(ctx_key_global, key, 32);
    encryptdemo(key, inputObjBuf, numCalls);


    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());

        assert(status == PIM_OK);
    }


   
    pimShowStats();
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) { 
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        if (((*inputObjBuf)[i_aes_block]->array[i_chunk] & 0xff) != bufIn[j]) {
            
            std::cout << "(int)(*inputObjBuf)[" << i_aes_block << "]->array[" << i_chunk << "]: " << (int)(*inputObjBuf)[i_aes_block]->array[i_chunk] << std::endl;
            std::cout << "buf[" << j << "]: " << (int)bufIn[j] << std::endl;
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testDecryptdemo(void) {
    std::cout << "INFO: PIM test: AES.decryptdemo" << std::endl;

    // unsigned long numBytes = 1310720; 
    unsigned long numBytes = 1UL * 1024 * 1024 * 1024; // 1 GB

    // Each rank has 8 chips; Total Bank = 16; Each Bank contains 32 subarrays;
    unsigned numRanks = 2;
    unsigned numBankPerRank = 128; // 8 chips * 16 banks
    unsigned numSubarrayPerBank = 32;
    unsigned numRows = 8192;
    unsigned numCols = 8192;
    unsigned numCores = numRanks * numBankPerRank * numSubarrayPerBank / 2; 
    unsigned totalCols = numCores * numCols;

    unsigned numCalls = 1;
    unsigned numPaddedBufBytes = numBytes;
    unsigned numElements = numPaddedBufBytes / AES_BLOCK_SIZE;


    {
        std::cout << "INFO: numCalls = " << numCalls << std::endl; 
        std::cout << "INFO: numPaddedBufBytes = " << numPaddedBufBytes << std::endl; 
        std::cout << std::endl;

    }
    
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
    assert(status == PIM_OK);


    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE * numCalls);
    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
   
    for (unsigned j = 1; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    uint8_t key[32];
    for (unsigned j = 0; j < 32; ++j)
    {
        key[j] = rand() % 256;
    }

    // Allocate the input buffer 
    uint8_t* bufIn = (uint8_t*) malloc(numPaddedBufBytes * sizeof(uint8_t)); 

    // Initialize buffer 
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) {
        bufIn[j] = rand() % 256; 
    }

    // Initialize inputObjBuf
    unsigned words_per_call, i_call, i_chunk, i_aes_block, remained; 
    words_per_call = (totalCols * AES_BLOCK_SIZE);
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) {
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        (*inputObjBuf)[i_aes_block]->array[i_chunk] = bufIn[j]; 
    }

    // Copy inputObjBuf to the device 
    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }
    
    memcpy(ctx_key_global, key, 32);
    decryptdemo(key, bufIn, numPaddedBufBytes);   

    memcpy(ctx_key_global, key, 32);
    decryptdemo(key, inputObjBuf, numCalls);

    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());

        assert(status == PIM_OK);
    }
   
    pimShowStats();
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) { 
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        if (((*inputObjBuf)[i_aes_block]->array[i_chunk] & 0xff) != bufIn[j]) {
            
            std::cout << "(int)(*inputObjBuf)[" << i_aes_block << "]->array[" << i_chunk << "]: " << ((int)(*inputObjBuf)[i_aes_block]->array[i_chunk] & 0xff) << std::endl;
            std::cout << "buf[" << j << "]: " << (int)bufIn[j] << std::endl;
            std::cout << "Abort" << std::endl;
            return 1; 
        }
    }
    std::cout << "INFO: All correct!" << std::endl;
    return 0;
}

int testDemo(int argc, char **argv) {
    struct Params params = getInputParams(argc, argv);
    
    FILE *file;
    uint8_t *buf;
    unsigned long numbytes;
    clock_t start, end;
    int padding;
    uint8_t key[32]; // Encryption/Decryption key.

    // Open and read the key file.
    bool generateRandomKey = false;
    if (params.keyFile == NULL) {
        generateRandomKey = true;
        printf("INFO: Key file is not specifed. Random key will be used.\n");
    }
    if (!generateRandomKey) {
        file = fopen(params.keyFile, "r");
        if (file == NULL) {
            printf("ERROR: Error opening key file %s\n", params.keyFile);
        }
    }

    // Read the key from the key file.
    if (!generateRandomKey && fread(key, 1, 32, file) != 32) {
        printf("ERROR: The key length in %s is not 32 characters\n", params.keyFile);
        fclose(file);
        return EXIT_FAILURE;
    } else {
        for (unsigned int i = 0; i < 32; ++i) {
            key[i] = rand() & 0xff;
        }
    }

    // Verify that there are no extra characters.
    if (!generateRandomKey) {
        char extra;
        if (fread(&extra, 1, 1, file) != 0) {
            printf("ERROR: The key length in %s is more than 32 characters\n", params.keyFile);
            fclose(file);
            return EXIT_FAILURE;
        }
        fclose(file);
    }
   
    // Open and read the input file.
    bool generateRandomInput = false;
    if (params.inputFile == NULL) {
        generateRandomInput = true;
        numbytes = params.inputSize;
        printf("INFO: Input file is not specifed. Random input will be used.\n");
    }
    if (!generateRandomInput) {
        file = fopen(params.inputFile, "r");
        if (file == NULL) {
            printf("ERROR: Error opening input file %s\n", params.inputFile);
        } 
    }
    if (!generateRandomInput) {
        fseek(file, 0L, SEEK_END);
        numbytes = ftell(file);
        fseek(file, 0L, SEEK_SET);
    }

    
    // Allocate memory for the file content.
    buf = (uint8_t*)malloc(numbytes * sizeof(uint8_t));
    if (buf == NULL) {
        printf("ERROR: Memory allocation error\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Read the file into the buffer.
    if (!generateRandomInput && fread(buf, 1, numbytes, file) != numbytes) {
        printf("ERROR: Unable to read all bytes from file %s\n", params.inputFile);
        fclose(file);
        free(buf);
        return EXIT_FAILURE;
    }
    if (!generateRandomInput) { 
        fclose(file);
    } else {
        for (unsigned int i = 0; i < params.inputSize; ++i) {
            buf[i] = rand() & 0xff;
        }
    }
  
    // generate padding
    padding = numbytes % AES_BLOCK_SIZE;
    numbytes += padding;
    printf("INFO: Padding file with %d bytes for a new size of %lu\n", padding, numbytes);

    // Each rank has 8 chips; Total Bank = 16; Each Bank contains 32 subarrays;
    unsigned numRanks = 2;
    unsigned numBankPerRank = 128; // 8 chips * 16 banks
    unsigned numSubarrayPerBank = 32;
    unsigned numRows = 8192;
    unsigned numCols = 8192;
    unsigned numCores = numRanks * numBankPerRank * numSubarrayPerBank / 2; 
    unsigned totalCols = numCores * numCols;

    unsigned numCalls = 1;
    unsigned numPaddedBufBytes = numbytes;
    unsigned numElements = numPaddedBufBytes / AES_BLOCK_SIZE;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
    assert(status == PIM_OK);

    std::vector<PIMAuxilary*> *inputObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE * numCalls);
    (*inputObjBuf)[0]= new PIMAuxilary(PIM_ALLOC_AUTO, numElements, PIM_UINT8);
    for (unsigned j = 1; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        (*inputObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
    }
    
    // Allocate the input buffer 
    uint8_t* bufIn = (uint8_t*) malloc(numbytes * sizeof(uint8_t)); 
    uint8_t* bufInPadded = (uint8_t*) malloc(numPaddedBufBytes * sizeof(uint8_t)); 

    // Initialize buffer 
    for (unsigned j = 0; j < numbytes; ++j) {
        bufIn[j] = buf[j];
    }

    // Pad the buffer with zero and update the pointer.
    bufferPadding(bufIn, bufInPadded, numbytes, numPaddedBufBytes);
    free(bufIn);
    bufIn = bufInPadded;

    // Initialize inputObjBuf
    unsigned words_per_call, i_call, i_chunk, i_aes_block, remained; 
    words_per_call = (totalCols * AES_BLOCK_SIZE);
    for (unsigned j = 0; j < numPaddedBufBytes; ++j) {
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        (*inputObjBuf)[i_aes_block]->array[i_chunk] = bufIn[j]; 
    }
    // Copy inputObjBuf to the device 
    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
        status = pimCopyHostToDevice((*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    memcpy(ctx_key_global, key, 32);
    encryptdemo(key, bufIn, numPaddedBufBytes);    
    
    memcpy(ctx_key_global, key, 32);
    encryptdemo(key, inputObjBuf, numCalls);
        
    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
            status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
            assert(status == PIM_OK);
    }
 
    pimShowStats();

    for (unsigned j = 0; j < numPaddedBufBytes; ++j) { 
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        if ((*inputObjBuf)[i_aes_block]->array[i_chunk] != bufIn[j]) {
            
            std::cout << "(int)(*inputObjBuf)[" << i_aes_block << "]->array[" << i_chunk << "]: " << (int)(*inputObjBuf)[i_aes_block]->array[i_chunk] << std::endl;
            std::cout << "buf[" << j << "]: " << (int)bufIn[j] << std::endl;
            std::cout << "ERROR: Abort" << std::endl;
            return 1; 
        }
    }
 
    // write the ciphertext to file
    file = fopen(params.cipherFile, "w");
    fwrite(buf, 1, numbytes, file);
    fclose(file);
   
    // return EXIT_SUCCESS;


    memcpy(ctx_key_global, key, 32);
    decryptdemo(key, bufIn, numPaddedBufBytes);
    
    memcpy(ctx_key_global, key, 32);
    decryptdemo(key, inputObjBuf, numCalls);

    pimShowStats();
          
    for (unsigned j = 0; j < (AES_BLOCK_SIZE * numCalls); ++j) {
            status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (*inputObjBuf)[j]->array.data());
            assert(status == PIM_OK);
    }

    for (unsigned j = 0; j < numPaddedBufBytes; ++j) { 
        i_call = j / words_per_call; 
        remained = j % words_per_call; 
        i_aes_block = (remained % AES_BLOCK_SIZE) + (i_call * AES_BLOCK_SIZE);
        i_chunk = remained / AES_BLOCK_SIZE;
        if ((*inputObjBuf)[i_aes_block]->array[i_chunk] != bufIn[j]) {
            
            std::cout << "(int)(*inputObjBuf)[" << i_aes_block << "]->array[" << i_chunk << "]: " << (int)(*inputObjBuf)[i_aes_block]->array[i_chunk] << std::endl;
            std::cout << "buf[" << j << "]: " << (int)bufIn[j] << std::endl;
            std::cout << "ERROR: Abort" << std::endl;
            return 1; 
        }
    }

    // write to file
    file = fopen(params.outputFile, "w");
    fwrite(buf, 1, numbytes - padding, file);
    fclose(file);

    // Compare input and output files
    if (params.shouldVerify) { 
        if (compare_files(params.inputFile, params.outputFile) == 0) {
            printf("INFO: The input file and the output file are the same.\n");
        } else {
            printf("ERROR: The input file and the output file are different.\n");
        }
    }

    free(buf);
    return EXIT_SUCCESS;
}

void rjXtime(PIMAuxilary* xObj, PIMAuxilary* returnValueObj){
    int status; 

    PIMAuxilary* shiftedObj = new PIMAuxilary(xObj);
    status = pimCopyDeviceToDevice(xObj->pimObjId, shiftedObj->pimObjId);
    assert(status == PIM_OK);
    // uint8_t shifted = x << 1;
    pimShiftBitsLeft(shiftedObj->pimObjId, shiftedObj->pimObjId, 1);

    // uint8_t mask = x;
    PIMAuxilary* maskObj = new PIMAuxilary(xObj);
    status = pimCopyDeviceToDevice(xObj->pimObjId, maskObj->pimObjId);
    assert(status == PIM_OK);
    
    uint8_t const1 = 0x80;

    // mask = mask & const1;
    // status = pimAnd(maskObj->pimObjId, const1Obj->pimObjId, maskObj->pimObjId);
    status = pimAndScalar(maskObj->pimObjId, maskObj->pimObjId, const1);
    assert(status == PIM_OK);
    
    // mask = mask >> 7;  
    pimShiftBitsRight(maskObj->pimObjId, maskObj->pimObjId, 7);

    /* TODO: Replace pimMul with 1-bit to 8-bit AND operation */
#ifdef RJXTIME_FUCNTIONAL
    pimMulScalar(maskObj->pimObjId, maskObj->pimObjId, 0x1b);
#else 
    pimAndScalar(maskObj->pimObjId, maskObj->pimObjId, 0x1b);
#endif

    pimXor(maskObj->pimObjId, shiftedObj->pimObjId, returnValueObj->pimObjId);
    assert (status == PIM_OK);

    pimFree(shiftedObj->pimObjId);
    pimFree(maskObj->pimObjId);
}

void aesSubBytes(std::vector<PIMAuxilary*>* inputObjBuf) {

    /* TODO: Implementation based on bit-serial look-up table */
#ifdef SUBBYTES_FUNCTIONAL
     int status;
     // Copy input buffer back to the host 
     for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
         status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
          assert(status == PIM_OK);
     }

     auto start = std::chrono::high_resolution_clock::now();
     uint8_t b;
     for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j)
     {
         #pragma omp parallel for 
         for (unsigned i = 0; i < (*inputObjBuf)[0]->numElements; ++i) {
             b = (*inputObjBuf)[j]->array[i];
             (*inputObjBuf)[j]->array[i] = sbox[b];
         }
     }
     auto end = std::chrono::high_resolution_clock::now();
     host_elapsedTime_global += (end - start);

     // Copy input buffer to the device 
     for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
         status = pimCopyHostToDevice((void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
         assert(status == PIM_OK);
     }
#else     
    int totalAndOperations = 318 * AES_BLOCK_SIZE / 8; 
    int orOperations = 415 * AES_BLOCK_SIZE / 8 ;
    int NotOperations = 8 * AES_BLOCK_SIZE / 8; 

    for (int i = 0; i < totalAndOperations; i++) {
        pimAnd((*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId);
    }

    for (int i = 0; i < orOperations; i++) {
        pimOr((*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId);
    }
#endif    
    /* END */

}

void aesSubBytesInv(std::vector<PIMAuxilary*>* inputObjBuf) {

    /* TODO: Implementation based on bit-serial look-up table */
#ifdef SUBBYTES_FUNCTIONAL
    int status;
    // Copy input buffer to the host 
        for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyDeviceToHost((*inputObjBuf)[j]->pimObjId, (void*)(*inputObjBuf)[j]->array.data());
        assert(status == PIM_OK);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    uint8_t b;
    #pragma omp parallel for 
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j)
    {
        for (unsigned i = 0; i < (*inputObjBuf)[0]->numElements; ++i) {
            b = (*inputObjBuf)[j]->array[i];
            (*inputObjBuf)[j]->array[i] = sboxinv[b];
        }   
    }
    auto end = std::chrono::high_resolution_clock::now();
    host_elapsedTime_global += (end - start);

    // Copy input buffer back to the device 
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        status = pimCopyHostToDevice((void*)(*inputObjBuf)[j]->array.data(), (*inputObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);       
    }

#else
    int totalAndOperations = 302 * AES_BLOCK_SIZE / 8; 
    int orOperations = 428 * AES_BLOCK_SIZE / 8 ;
    int NotOperations = 37 * AES_BLOCK_SIZE / 8; 

    for (int i = 0; i < totalAndOperations; i++) {
        pimAnd((*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId);
    }

    for (int i = 0; i < orOperations; i++) {
        pimOr((*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId, (*inputObjBuf)[0]->pimObjId);
    }

   
#endif
    /* END */
}

void aesAddRoundKey(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* outObjBuf) {
    int status;
    for (int j = 0; j < AES_BLOCK_SIZE; j++){ 
        pimXor((*inputObjBuf)[j]->pimObjId, (*keyObjBuf)[j]->pimObjId, (*outObjBuf)[j]->pimObjId); 
    }   
}

void aesAddRoundKeyCpy(std::vector<PIMAuxilary*>* inputObjBuf,std::vector<PIMAuxilary*>* keyObjBuf, std::vector<PIMAuxilary*>* cpkObjBuf, std::vector<PIMAuxilary*>* outObjBuf) {
int status;

    /* for (int j = 0; j < 16; j++){ 
       cpk[j] = key[j];
    } */
    for (int j = 0; j < 16; j++){ 
        pimCopyDeviceToDevice((*keyObjBuf)[j]->pimObjId, (*cpkObjBuf)[j]->pimObjId);
    } 

    /* for (int j = 0; j < 16; j++){ 
       buf[j] ^= cpk[j];
    } */
    for (int j = 0; j < 16; j++){ 
        pimXor((*inputObjBuf)[j]->pimObjId, (*cpkObjBuf)[j]->pimObjId, (*outObjBuf)[j]->pimObjId); 
    }   

    /* for (int j = 0; j < 16; j++){ 
       cpk[16 + j] = key[16 + j];
    } */
    for (int j = 0; j < 16; j++){ 
        pimCopyDeviceToDevice((*keyObjBuf)[16 + j]->pimObjId, (*cpkObjBuf)[16 + j]->pimObjId);
    } 

    
}

void aesShiftRows(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status;
    PIMAuxilary* iObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* jObj = new PIMAuxilary((*inputObjBuf)[0]);
 
    pimCopyDeviceToDevice((*inputObjBuf)[1]->pimObjId, iObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[5]->pimObjId, (*inputObjBuf)[1]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[9]->pimObjId, (*inputObjBuf)[5]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[13]->pimObjId, (*inputObjBuf)[9]->pimObjId);
    pimCopyDeviceToDevice(iObj->pimObjId, (*inputObjBuf)[13]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[10]->pimObjId, iObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[2]->pimObjId, (*inputObjBuf)[10]->pimObjId);
    pimCopyDeviceToDevice(iObj->pimObjId, (*inputObjBuf)[2]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[3]->pimObjId, jObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[15]->pimObjId, (*inputObjBuf)[3]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[11]->pimObjId, (*inputObjBuf)[15]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[7]->pimObjId, (*inputObjBuf)[11]->pimObjId);
    pimCopyDeviceToDevice(jObj->pimObjId, (*inputObjBuf)[7]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[14]->pimObjId, jObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[6]->pimObjId, (*inputObjBuf)[14]->pimObjId);
    pimCopyDeviceToDevice(jObj->pimObjId, (*inputObjBuf)[6]->pimObjId);

    pimFree(iObj->pimObjId);
    pimFree(jObj->pimObjId);

}

void aesShiftRowsInv(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status;
    PIMAuxilary* iObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* jObj = new PIMAuxilary((*inputObjBuf)[0]);



    pimCopyDeviceToDevice((*inputObjBuf)[1]->pimObjId, iObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[13]->pimObjId, (*inputObjBuf)[1]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[9]->pimObjId, (*inputObjBuf)[13]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[5]->pimObjId, (*inputObjBuf)[9]->pimObjId);
    pimCopyDeviceToDevice(iObj->pimObjId, (*inputObjBuf)[5]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[2]->pimObjId, iObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[10]->pimObjId, (*inputObjBuf)[2]->pimObjId);
    pimCopyDeviceToDevice(iObj->pimObjId, (*inputObjBuf)[10]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[3]->pimObjId, jObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[7]->pimObjId, (*inputObjBuf)[3]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[11]->pimObjId, (*inputObjBuf)[7]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[15]->pimObjId, (*inputObjBuf)[11]->pimObjId);
    pimCopyDeviceToDevice(jObj->pimObjId, (*inputObjBuf)[15]->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[6]->pimObjId, jObj->pimObjId);
    pimCopyDeviceToDevice((*inputObjBuf)[14]->pimObjId, (*inputObjBuf)[6]->pimObjId);
    pimCopyDeviceToDevice(jObj->pimObjId, (*inputObjBuf)[14]->pimObjId);

    pimFree(iObj->pimObjId);
    pimFree(jObj->pimObjId);
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
    PIMAuxilary* tObj = new PIMAuxilary((*inputObjBuf)[0]);

    for (int j = 0; j < 16; j += 4){
        //  a = buf[j];
        pimCopyDeviceToDevice((*inputObjBuf)[j]->pimObjId, aObj->pimObjId);

        // b = buf[j + 1];
        pimCopyDeviceToDevice((*inputObjBuf)[j + 1]->pimObjId, bObj->pimObjId);

        // c = buf[j + 2];
        pimCopyDeviceToDevice((*inputObjBuf)[j + 2]->pimObjId, cObj->pimObjId);

        // d = buf[j + 3];
        pimCopyDeviceToDevice((*inputObjBuf)[j + 3]->pimObjId, dObj->pimObjId);
        
        // e = a ^ b;
        pimXor(aObj->pimObjId, bObj->pimObjId, eObj->pimObjId); 

        // t = e ^ c;
        pimXor(eObj->pimObjId, cObj->pimObjId, tObj->pimObjId); 

        // e = t ^ d;
        pimXor(tObj->pimObjId, dObj->pimObjId, eObj->pimObjId); 
             
        // t = a ^ b;
        pimXor(aObj->pimObjId, bObj->pimObjId, tObj->pimObjId); 

        // f = rj_xtime(t);
        rjXtime(tObj, fObj);

        // t = f ^ e; 
        pimXor(fObj->pimObjId, eObj->pimObjId, tObj->pimObjId); 

        // f = buf[j] ^ t;
        pimXor((*inputObjBuf)[j]->pimObjId, tObj->pimObjId, fObj->pimObjId);

        // buf[j] = f
        pimCopyDeviceToDevice(fObj->pimObjId, (*inputObjBuf)[j]->pimObjId);


        // t = b ^ c; 
        pimXor(bObj->pimObjId, cObj->pimObjId, tObj->pimObjId);

        // b = rj_xtime(t);
        rjXtime(tObj, bObj);

        // t = b ^ e; 
        pimXor(bObj->pimObjId, eObj->pimObjId, tObj->pimObjId);


        // b =  buf[j+1] ^ t;
        pimXor((*inputObjBuf)[j + 1]->pimObjId, tObj->pimObjId, bObj->pimObjId);   

        // buf[j+1] = b
        pimCopyDeviceToDevice(bObj->pimObjId, (*inputObjBuf)[j + 1]->pimObjId);

        // t = c ^ d; 
        pimXor(cObj->pimObjId, dObj->pimObjId, tObj->pimObjId);
        
        // c = rj_xtime(t);
        rjXtime(tObj, cObj);

        // t = c ^ e; 
        pimXor(cObj->pimObjId, eObj->pimObjId, tObj->pimObjId);

        // c = buf[j+2] ^ t;
        pimXor((*inputObjBuf)[j + 2]->pimObjId, tObj->pimObjId, cObj->pimObjId);

        // buf[j+2] = c
        pimCopyDeviceToDevice(cObj->pimObjId, (*inputObjBuf)[j + 2]->pimObjId);

        // t = d ^ a; 
        pimXor(dObj->pimObjId, aObj->pimObjId, tObj->pimObjId);

        // d = rj_xtime(t);
        rjXtime(tObj, dObj);

        // t = d ^ e; 
        pimXor(dObj->pimObjId, eObj->pimObjId, tObj->pimObjId);

        // d = buf[j+3] ^ t;
        pimXor((*inputObjBuf)[j + 3]->pimObjId, tObj->pimObjId, dObj->pimObjId);

        // buf[j+3] = d
        pimCopyDeviceToDevice(dObj->pimObjId, (*inputObjBuf)[j + 3]->pimObjId);

    }
    pimFree(aObj->pimObjId);
    pimFree(bObj->pimObjId);
    pimFree(cObj->pimObjId);
    pimFree(dObj->pimObjId);
    pimFree(eObj->pimObjId);
    pimFree(fObj->pimObjId);
    pimFree(tObj->pimObjId);
}       


void aesMixColumnsInv(std::vector<PIMAuxilary*>* inputObjBuf) {
    int status; 
    
    // uint8_t a, b, c, d, e, x, y, z, t0;
    PIMAuxilary* aObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* bObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* cObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* dObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* eObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* xObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* yObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* zObj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* t0Obj = new PIMAuxilary((*inputObjBuf)[0]);
    PIMAuxilary* tObj = new PIMAuxilary((*inputObjBuf)[0]);
 

    for (int j = 0; j < 16; j += 4){
        // a = buf[j];
        pimCopyDeviceToDevice((*inputObjBuf).data()[j]->pimObjId, aObj->pimObjId);
        // aObj->pimObjId = (*inputObjBuf).data()[j]->pimObjId;

        // b = buf[j + 1];
        pimCopyDeviceToDevice((*inputObjBuf).data()[j + 1]->pimObjId, bObj->pimObjId);

        // c = buf[j + 2];
        pimCopyDeviceToDevice((*inputObjBuf).data()[j + 2]->pimObjId, cObj->pimObjId);

        // d = buf[j + 3];
        pimCopyDeviceToDevice((*inputObjBuf).data()[j + 3]->pimObjId, dObj->pimObjId);
              
        // e = a ^ b;
        pimXor(aObj->pimObjId, bObj->pimObjId, eObj->pimObjId);


        // t = e ^ c;
        pimXor(eObj->pimObjId, cObj->pimObjId, tObj->pimObjId);
            
        // e = t ^ d;
        pimXor(tObj->pimObjId, dObj->pimObjId, eObj->pimObjId);

        // z = rj_xtime(e);
        rjXtime(eObj, zObj);


        // t = a ^ c;
        pimXor(aObj->pimObjId, cObj->pimObjId, tObj->pimObjId);
        
        // t0 = t ^ z;
        pimXor(tObj->pimObjId, zObj->pimObjId, t0Obj->pimObjId);

        // t = rj_xtime(t0);
        rjXtime(t0Obj, tObj);

        // t0 = rj_xtime(t);
        rjXtime(tObj, t0Obj);
    
        // x = e ^ t0;
        pimXor(t0Obj->pimObjId, eObj->pimObjId, xObj->pimObjId);
             
        
        // t = b ^ d;
        pimXor(bObj->pimObjId, dObj->pimObjId, tObj->pimObjId);

        // t0 = t ^ z;
        pimXor(tObj->pimObjId, zObj->pimObjId, t0Obj->pimObjId);

        // t = rj_xtime(t0);
        rjXtime(t0Obj, tObj);

        // t0 = rj_xtime(t);
        rjXtime(tObj, t0Obj);

        // y = e ^ t0;
        pimXor(t0Obj->pimObjId, eObj->pimObjId, yObj->pimObjId);


        // t0 = a ^ b;
        pimXor(aObj->pimObjId, bObj->pimObjId, t0Obj->pimObjId);

        // t0 = rj_xtime(t0);
        rjXtime(t0Obj, t0Obj);

        // t = t0 ^ x;
        pimXor(t0Obj->pimObjId, xObj->pimObjId, tObj->pimObjId);
        
        // t0 = buf[j] ^ t;
        pimXor((*inputObjBuf)[j]->pimObjId, tObj->pimObjId, t0Obj->pimObjId);

        // buf[j] = t0;
        pimCopyDeviceToDevice(t0Obj->pimObjId, (*inputObjBuf)[j]->pimObjId);


        // t0 = b ^ c;
        pimXor(bObj->pimObjId, cObj->pimObjId, t0Obj->pimObjId);

        // t = rj_xtime(t0);
        rjXtime(t0Obj, tObj);

        // t0 = t ^ y;
        pimXor(tObj->pimObjId, yObj->pimObjId, t0Obj->pimObjId);

        // t = buf[j + 1] ^ t0;
        pimXor((*inputObjBuf)[j + 1]->pimObjId, t0Obj->pimObjId, tObj->pimObjId);

        // buf[j + 1] = t;
        pimCopyDeviceToDevice(tObj->pimObjId, (*inputObjBuf)[j + 1]->pimObjId);

 
        // t0 = c ^ d;
        pimXor(cObj->pimObjId, dObj->pimObjId, t0Obj->pimObjId);

        // t = rj_xtime(t0);
        rjXtime(t0Obj, tObj);

        // t0 = t ^ x;
        pimXor(tObj->pimObjId, xObj->pimObjId, t0Obj->pimObjId);

        // t = buf[j + 2] ^ t0;
        pimXor((*inputObjBuf)[j + 2]->pimObjId, t0Obj->pimObjId, tObj->pimObjId);

        // buf[j + 2] = t;
        pimCopyDeviceToDevice(tObj->pimObjId, (*inputObjBuf)[j + 2]->pimObjId);


        // t0 = d ^ a;
        pimXor(dObj->pimObjId, aObj->pimObjId, t0Obj->pimObjId);

        // t = rjXtime(t0);
        rjXtime(t0Obj, tObj);

        // t0 = t ^ y;
        pimXor(tObj->pimObjId, yObj->pimObjId, t0Obj->pimObjId);

        // t = buf[j + 3] ^ t0;
        pimXor((*inputObjBuf)[j + 3]->pimObjId, t0Obj->pimObjId, tObj->pimObjId);

        // buf[j + 3] = t;
        pimCopyDeviceToDevice(tObj->pimObjId, (*inputObjBuf)[j + 3]->pimObjId);
    }
    pimFree(aObj->pimObjId);
    pimFree(bObj->pimObjId);
    pimFree(cObj->pimObjId);
    pimFree(dObj->pimObjId);
    pimFree(eObj->pimObjId);
    pimFree(xObj->pimObjId);
    pimFree(yObj->pimObjId);
    pimFree(zObj->pimObjId);
    pimFree(t0Obj->pimObjId);
    pimFree(tObj->pimObjId);
}       

// add expand key operation
void aesExpandEncKey(uint8_t *k, uint8_t *rc, const uint8_t *sb){
    uint8_t i;

    k[0] ^= sb[k[29]] ^ (*rc);
    k[1] ^= sb[k[30]];
    k[2] ^= sb[k[31]];
    k[3] ^= sb[k[28]];
    *rc = F( *rc);

    for(i = 4; i < 16; i += 4){
        k[i] ^= k[i-4];
        k[i+1] ^= k[i-3];
        k[i+2] ^= k[i-2];
        k[i+3] ^= k[i-1];
    }

    k[16] ^= sb[k[12]];
    k[17] ^= sb[k[13]];
    k[18] ^= sb[k[14]];
    k[19] ^= sb[k[15]];

    for(i = 20; i < 32; i += 4){
        k[i] ^= k[i-4];
        k[i+1] ^= k[i-3];
        k[i+2] ^= k[i-2];
        k[i+3] ^= k[i-1];
    }
}

// inv add expand key operation
void aesExpandDecKey(uint8_t *k, uint8_t *rc){
    uint8_t i;

    for(i = 28; i > 16; i -= 4){
        k[i+0] ^= k[i-4];
        k[i+1] ^= k[i-3];
        k[i+2] ^= k[i-2];
        k[i+3] ^= k[i-1];
    }

    k[16] ^= sbox[k[12]];
    k[17] ^= sbox[k[13]];
    k[18] ^= sbox[k[14]];
    k[19] ^= sbox[k[15]];

    for(i = 12; i > 0; i -= 4){
        k[i+0] ^= k[i-4];
        k[i+1] ^= k[i-3];
        k[i+2] ^= k[i-2];
        k[i+3] ^= k[i-1];
    }

    *rc = FD(*rc);
    k[0] ^= sbox[k[29]] ^ (*rc);
    k[1] ^= sbox[k[30]];
    k[2] ^= sbox[k[31]];
    k[3] ^= sbox[k[28]];
} 

// key initition
void aes256Init(uint8_t *k){
  uint8_t rcon = 1;
  uint8_t i;

  for (i = 0; i < sizeof(ctx_deckey_global); i++){
    ctx_enckey_global[i] = ctx_deckey_global[i] = k[i];
  }

  for (i = 8; --i; ){
    aesExpandEncKey(ctx_deckey_global, &rcon, sbox);
  }
} 

void aes256EncryptEcb(std::vector<PIMAuxilary*>* inputObjBuf, unsigned long offset) {
    int status;
    uint8_t l = 1, rcon;

    uint8_t ctx_enckey_local[32];
    uint8_t ctx_key_local[32];
    memcpy(ctx_enckey_local, ctx_enckey_global, 32);
    memcpy(ctx_key_local, ctx_key_global, 32);

    // uint8_t bufT[AES_BLOCK_SIZE];
    std::vector<PIMAuxilary*> *bufTObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*bufTObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[j + offset]);
        pimCopyDeviceToDevice((*inputObjBuf)[j + offset]->pimObjId, (*bufTObjBuf)[j]->pimObjId);
    }

    std::vector<PIMAuxilary*> *bufT0ObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*bufT0ObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[j + offset]);
        pimCopyDeviceToDevice((*inputObjBuf)[j + offset]->pimObjId, (*bufT0ObjBuf)[j]->pimObjId);
    }

    std::vector<PIMAuxilary*>* keyObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*keyObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*>* cpkObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*cpkObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    for (unsigned j = 0; j < 32; ++j) {
        pimBroadcastUInt((*keyObjBuf)[j]->pimObjId, ctx_enckey_local[j]);
        pimBroadcastUInt((*cpkObjBuf)[j]->pimObjId, ctx_key_local[j]);
    }

    aesAddRoundKeyCpy(bufTObjBuf, keyObjBuf, cpkObjBuf, bufT0ObjBuf);
    
    for(l = 1, rcon = l; l < 14; ++l){
        aesSubBytes(bufT0ObjBuf);
        aesShiftRows(bufT0ObjBuf); 
        aesMixColumns(bufT0ObjBuf);

        if( l & 1 ){
            aesAddRoundKey(bufT0ObjBuf, cpkObjBuf, bufTObjBuf);
        }
        else{
            aesExpandEncKey(ctx_key_local, &rcon, sbox);
            for (unsigned j = 0; j < 32; ++j) {
                status = pimBroadcastUInt((*cpkObjBuf)[j]->pimObjId, ctx_key_local[j]);
                assert (status == PIM_OK);
            }
            aesAddRoundKey(bufT0ObjBuf, cpkObjBuf, bufTObjBuf);
        }
        for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
            pimCopyDeviceToDevice((*bufTObjBuf)[j]->pimObjId, (*bufT0ObjBuf)[j]->pimObjId);
        }
    }
    aesSubBytes(bufTObjBuf);
    aesShiftRows(bufTObjBuf);
    aesExpandEncKey(ctx_key_local, &rcon, sbox);
    for (unsigned j = 0; j < 32; ++j) {
        status = pimBroadcastUInt((*cpkObjBuf)[j]->pimObjId, ctx_key_local[j]);
        assert(status == PIM_OK);
    }
    aesAddRoundKey(bufTObjBuf, cpkObjBuf, bufT0ObjBuf);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        pimCopyDeviceToDevice((*bufT0ObjBuf)[j]->pimObjId, (*inputObjBuf)[j + offset]->pimObjId);
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        pimFree((*bufTObjBuf)[j]->pimObjId);
        pimFree((*bufT0ObjBuf)[j]->pimObjId);
    }
    for (unsigned j = 0; j < 32; ++j) {
        pimFree((*keyObjBuf)[j]->pimObjId);
        pimFree((*cpkObjBuf)[j]->pimObjId);
    }
}

void aes256DecryptEcb(std::vector<PIMAuxilary*>* inputObjBuf, unsigned long offset) {
    int status;
    uint8_t l, rcon;

    uint8_t ctx_deckey_local[32];
    uint8_t ctx_key_local[32];
    uint8_t bufT[AES_BLOCK_SIZE];

    memcpy(ctx_deckey_local, ctx_deckey_global, 32);
    memcpy(ctx_key_local, ctx_key_global, 32);
        
    std::vector<PIMAuxilary*> *bufTObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*bufTObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[j + offset]);
        status = pimCopyDeviceToDevice((*inputObjBuf)[j + offset]->pimObjId, (*bufTObjBuf)[j]->pimObjId);
        // status = pimCopyHostToDevice((void*)(*inputObjBuf)[j + offset]->array.data(), (*bufTObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    std::vector<PIMAuxilary*> *bufT0ObjBuf = new std::vector<PIMAuxilary*>(AES_BLOCK_SIZE);
    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        (*bufT0ObjBuf)[j]= new PIMAuxilary((*inputObjBuf)[0]);
        // status = pimCopyHostToDevice((void*)(*inputObjBuf)[j + offset]->array.data(), (*bufT0ObjBuf)[j]->pimObjId);
        status = pimCopyDeviceToDevice((*inputObjBuf)[j + offset]->pimObjId, (*bufT0ObjBuf)[j]->pimObjId);
        assert(status == PIM_OK);
    }

    std::vector<PIMAuxilary*>* keyObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*keyObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    std::vector<PIMAuxilary*>* cpkObjBuf = new std::vector<PIMAuxilary*>(32);
    for (unsigned j = 0; j < 32; ++j) {
        (*cpkObjBuf)[j] = new PIMAuxilary((*inputObjBuf)[0]);
    }

    for (unsigned j = 0; j < 32; ++j) {
        pimBroadcastUInt((*keyObjBuf)[j]->pimObjId, ctx_deckey_local[j]);
        pimBroadcastUInt((*cpkObjBuf)[j]->pimObjId, ctx_key_local[j]);
    }

    aesAddRoundKeyCpy(bufTObjBuf, keyObjBuf, cpkObjBuf, bufT0ObjBuf);
    aesAddRoundKeyCpy(bufT, ctx_deckey_local, ctx_key_local);

    aesShiftRowsInv(bufT0ObjBuf);
    aesSubBytesInv(bufT0ObjBuf);

    for (l = 14, rcon = 0x80; --l;){
        if((l & 1)){
   
            aesExpandDecKey(ctx_key_local, &rcon);
            for (unsigned j = 0; j < 32; ++j) {
                pimBroadcastUInt((*cpkObjBuf)[j]->pimObjId, ctx_key_local[j]);
            }
            // aesAddRoundKey(bufT, ctx_key_local);
            aesAddRoundKey(bufT0ObjBuf, cpkObjBuf, bufT0ObjBuf);

        }
        else{
            // aesAddRoundKey(bufT, ctx_key_local);
            aesAddRoundKey(bufT0ObjBuf, cpkObjBuf, bufT0ObjBuf);
        }

        // aesMixColumnsInv(bufT);
        aesMixColumnsInv(bufT0ObjBuf);

        // aesShiftRowsInv(bufT);
        aesShiftRowsInv(bufT0ObjBuf);
        
        // aesSubBytesInv(bufT);
        aesSubBytesInv(bufT0ObjBuf);
    }
    aesAddRoundKey(bufT0ObjBuf, cpkObjBuf, bufTObjBuf);

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        pimCopyDeviceToDevice((*bufTObjBuf)[j]->pimObjId, (*inputObjBuf)[j + offset]->pimObjId);
    }

    for (unsigned j = 0; j < AES_BLOCK_SIZE; ++j) {
        pimFree((*bufTObjBuf)[j]->pimObjId);
        pimFree((*bufT0ObjBuf)[j]->pimObjId);
    }
    for (unsigned j = 0; j < 32; ++j) {
        pimFree((*keyObjBuf)[j]->pimObjId);
        pimFree((*cpkObjBuf)[j]->pimObjId);
    }
}

void encryptdemo(uint8_t key[32], std::vector<PIMAuxilary*>* inputObjBuf, unsigned long numCalls) {
    aes256Init(key);
    unsigned long offset = 0;
    
    for (int  pass = 0; pass < numCalls; ++pass) {
        aes256EncryptEcb(inputObjBuf, offset);
        offset += AES_BLOCK_SIZE;
    }
}

void decryptdemo(uint8_t key[32], std::vector<PIMAuxilary*>* inputObjBuf, unsigned long numCalls) {
    aes256Init(key);
    unsigned long offset = 0;

    for (int  pass = 0; pass < numCalls; ++pass) {
        aes256DecryptEcb(inputObjBuf, offset);
        offset += AES_BLOCK_SIZE;
    }
}

int compare_files(const char *file1, const char *file2) {
    FILE *f1 = fopen(file1, "r");
    FILE *f2 = fopen(file2, "r");
    if (f1 == NULL || f2 == NULL) {
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        return -1;
    }

    int ch1, ch2;
    do {
        ch1 = fgetc(f1);
        ch2 = fgetc(f2);
        if (ch1 != ch2) {
            fclose(f1);
            fclose(f2);
            return -1;
        }
    } while (ch1 != EOF && ch2 != EOF);

    fclose(f1);
    fclose(f2);

    if (ch1 == EOF && ch2 == EOF) {
        return 0;
    } else {
        return -1;
    }
}

void usage() {
    fprintf(stderr,
        "\nUsage:  ./aes.out [options]"
        "\n"
        "\n    -l    input size (default=65536 bytes)"
        "\n    -k    key file containing AES key (default=generates key with random numbers)"
        "\n    -i    input file containing AES encrption input(default=generates input with random numbers)"
        "\n    -c    cipher file containing AES encryption output (default=./cipher.txt)"
        "\n    -o    output file containing AES decryption output (default=./output.txt)"
        "\n    -v    (true/false) validates if the input file and outputfile are the same. (default=false)"
        "\n");
}

struct Params getInputParams(int argc, char **argv) {
    struct Params p = {65536, NULL, NULL, "./cipher.txt", "./output.txt", false};
    int opt;

    while ((opt = getopt(argc, argv, "hl:k:i:c:o:v:")) >= 0) {
        switch (opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'l':
                p.inputSize = strtoull(optarg, NULL, 0);
                break;
            case 'k':
                p.keyFile = optarg;
                break;
            case 'i':
                p.inputFile = optarg;
                break;
            case 'c':
                p.cipherFile = optarg;
                break;
            case 'o':
                p.outputFile = optarg;
                break;
            case 'v':
                p.shouldVerify = (*optarg == 't') ? true : false;
                break;
            default:
                fprintf(stderr, "\nERROR: Unrecognized option!\n");
                usage();
                exit(0);
        }
    }
    return p;
}
 
// Util functions 
void inline interrupt(int line) { 
    std::cout << "DEBUG: Interrupted at line " << line << ". \nPress any key to continue ..." << std::endl;
    int dummy; std::cin >> dummy; 
} 
void bufferPadding(uint8_t* inBuf, uint8_t* outBuf, unsigned inNumBytes, unsigned outNumBytes) { 
    // Copy the occupied part of the input buffer to the output buffer
    memcpy(outBuf, inBuf, inNumBytes);

    // Zero the second part of the output buffer. 
    unsigned numZeroBytes = outNumBytes - inNumBytes;
    memset((void*) (outBuf + inNumBytes), 0, numZeroBytes * sizeof(uint8_t));
}


