#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#define MEASUREMENT_TIMES 10


uint8_t ctx_key[32]; 
uint8_t ctx_enckey[32]; 
uint8_t ctx_deckey[32];


#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512


#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))


// S table
__constant__ static const uint8_t sbox[256] = {
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


// inv S table
__constant__ static const uint8_t sboxinv[256] = {
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


// x-time operation
__device__ uint8_t rj_xtime(uint8_t x){
  return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}


// subbyte operation
__device__ void aes_subBytes(uint8_t *buf){
  register uint8_t i, b;
  for (i = 0; i < 16; ++i){
    b = buf[i];
    buf[i] = sbox[b];
  }
} 


// inv subbyte operation
__device__ void aes_subBytes_inv(uint8_t *buf){
  register uint8_t i, b;
  for (i = 0; i < 16; ++i){
    b = buf[i];
    buf[i] = sboxinv[b];
  }
} 


// add round key operation
__device__ void aes_addRoundKey(uint8_t *buf, uint8_t *key){
  register uint8_t i = 16;
  while (i--){
    buf[i] ^= key[i];
  }
} 


// add round key at beginning
__device__ void aes_addRoundKey_cpy(uint8_t *buf, uint8_t *key, uint8_t *cpk){
  register uint8_t i = 16;
  while (i--){
    buf[i] ^= (cpk[i] = key[i]);
    cpk[16+i] = key[16 + i];
  }
} 



// shift row operation
__device__ void aes_shiftRows(uint8_t *buf){
  register uint8_t i, j; 
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



// inv shift row operation
__device__ void aes_shiftRows_inv(uint8_t *buf){
  register uint8_t i, j; 
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


// mix column operation
__device__ void aes_mixColumns(uint8_t *buf){
  register uint8_t i, a, b, c, d, e;
  for (i = 0; i < 16; i += 4){
    a = buf[i];
    b = buf[i + 1];
    c = buf[i + 2];
    d = buf[i + 3];
    e = a ^ b ^ c ^ d;
    buf[i] ^= e ^ rj_xtime(a^b);
    buf[i+1] ^= e ^ rj_xtime(b^c);
    buf[i+2] ^= e ^ rj_xtime(c^d);
    buf[i+3] ^= e ^ rj_xtime(d^a);
  }
} 


// inv mix column operation
__device__ void aes_mixColumns_inv(uint8_t *buf){
  register uint8_t i, a, b, c, d, e, x, y, z;
  for (i = 0; i < 16; i += 4){
    a = buf[i];
    b = buf[i + 1];
    c = buf[i + 2];
    d = buf[i + 3];
    e = a ^ b ^ c ^ d;
    z = rj_xtime(e);
    x = e ^ rj_xtime(rj_xtime(z^a^c));
    y = e ^ rj_xtime(rj_xtime(z^b^d));
    buf[i] ^= x ^ rj_xtime(a^b);
    buf[i+1] ^= y ^ rj_xtime(b^c);
    buf[i+2] ^= x ^ rj_xtime(c^d);
    buf[i+3] ^= y ^ rj_xtime(d^a);
  }
} 


// add expand key operation
__device__ __host__ void aes_expandEncKey(uint8_t *k, uint8_t *rc, const uint8_t *sb){
  register uint8_t i;

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
__device__ void aes_expandDecKey(uint8_t *k, uint8_t *rc){
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
void aes256_init(uint8_t *k){
  uint8_t rcon = 1;
  register uint8_t i;

  for (i = 0; i < sizeof(ctx_key); i++){
    ctx_enckey[i] = ctx_deckey[i] = k[i];
  }
  for (i = 8;--i;){
    aes_expandEncKey(ctx_deckey, &rcon, sbox);
  }
} 



// aes encrypt algorithm one thread/one block with AES_BLOCK_SIZE 
__global__ void aes256_encrypt_ecb(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_enckey_d, uint8_t *ctx_key_d){
  uint8_t i, rcon;
  uint8_t buf_t[AES_BLOCK_SIZE]; // thread buffer
  //printf("Thread %d\n", threadIdx.x);
  unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
  if (offset >= numbytes) {  return; }

  memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);
  aes_addRoundKey_cpy(buf_t, ctx_enckey_d, ctx_key_d);
  for(i = 1, rcon = 1; i < 14; ++i){
    aes_subBytes(buf_t);
    aes_shiftRows(buf_t);
    aes_mixColumns(buf_t);
    if( i & 1 ){
      aes_addRoundKey( buf_t, &ctx_key_d[16]);
    }
    else{
      aes_expandEncKey(ctx_key_d, &rcon, sbox), aes_addRoundKey(buf_t, ctx_key_d);
    }
  }
  aes_subBytes(buf_t);
  aes_shiftRows(buf_t);
  aes_expandEncKey(ctx_key_d, &rcon, sbox);
  aes_addRoundKey(buf_t, ctx_key_d);
  /* copy thread buffer back into global memory */
  memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
  __syncthreads();
} 



// aes decrypt algorithm
__global__ void aes256_decrypt_ecb(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_deckey_d, uint8_t *ctx_key_d){
  uint8_t i, rcon;
  uint8_t buf_t[AES_BLOCK_SIZE];
  unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
  if (offset >= numbytes) { return; }
  memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);
  aes_addRoundKey_cpy(buf_t, ctx_deckey_d, ctx_key_d);
  aes_shiftRows_inv(buf_t);
  aes_subBytes_inv(buf_t);
  for (i = 14, rcon = 0x80; --i;){
    if( ( i & 1 ) ){
      aes_expandDecKey(ctx_key_d, &rcon);
      aes_addRoundKey(buf_t, &ctx_key_d[16]);
    }
    else{
      aes_addRoundKey(buf_t, ctx_key_d);
    }
    aes_mixColumns_inv(buf_t);
    aes_shiftRows_inv(buf_t);
    aes_subBytes_inv(buf_t);
    }
  aes_addRoundKey( buf_t, ctx_key_d);
  /* copy thread back into global memory */
  memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
  __syncthreads();
} 



// aes encrypt demo
void encryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;
  uint8_t *ctx_key_d, *ctx_enckey_d;

  cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t)*256);

  printf("\nBeginning encryption\n");
  aes256_init(key);

  cudaMalloc((void**)&buf_d, numbytes);
  cudaMalloc((void**)&ctx_enckey_d, sizeof(ctx_enckey));
  cudaMalloc((void**)&ctx_key_d, sizeof(ctx_key));

  cudaMemcpy(buf_d, buf, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx_enckey_d, ctx_enckey, sizeof(ctx_enckey), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx_key_d, ctx_key, sizeof(ctx_key), cudaMemcpyHostToDevice);
  dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  // printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);
  aes256_encrypt_ecb<<<dimBlock, dimGrid>>>(buf_d, numbytes, ctx_enckey_d, ctx_key_d);

  cudaMemcpy(buf, buf_d, numbytes, cudaMemcpyDeviceToHost);
  // print(buf);
  cudaMemcpy(ctx_enckey, ctx_enckey_d, sizeof(ctx_enckey), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx_key, ctx_key_d, sizeof(ctx_key), cudaMemcpyDeviceToHost);

  cudaFree(buf_d);
  cudaFree(ctx_key_d);
  cudaFree(ctx_enckey_d);
}


// aes decrypt demo
void decryptdemo(uint8_t key[32], uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;
  uint8_t *ctx_key_d, *ctx_deckey_d;

  cudaMemcpyToSymbol(sboxinv, sboxinv, sizeof(uint8_t)*256);

  printf("\nBeginning decryption\n");

  cudaMalloc((void**)&buf_d, numbytes);
  cudaMalloc((void**)&ctx_deckey_d, sizeof(ctx_deckey));
  cudaMalloc((void**)&ctx_key_d, sizeof(ctx_key));

  cudaMemcpy(buf_d, buf, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx_deckey_d, ctx_deckey, sizeof(ctx_deckey), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx_key_d, ctx_key, sizeof(ctx_key), cudaMemcpyHostToDevice);

  dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);
  aes256_decrypt_ecb<<<dimBlock, dimGrid>>>(buf_d, numbytes, ctx_deckey_d, ctx_key_d);

  cudaMemcpy(buf, buf_d, numbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx_deckey, ctx_deckey_d, sizeof(ctx_deckey), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx_key, ctx_key_d, sizeof(ctx_key), cudaMemcpyDeviceToHost);

  cudaFree(buf_d);
  cudaFree(ctx_key_d);
  cudaFree(ctx_deckey_d);
}


__global__ void GPU_init() { }



int main(){

  // open file
  FILE *file;
  uint8_t *buf; 
  unsigned long numbytes;
  char *fname;
  clock_t start, enc_time, dec_time, end;
  int mili_sec, i;
  int padding;
 
  uint8_t key[32];

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(error_id));
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0){
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }


  // handle txt file
  fname = "../input.txt";  
  file = fopen(fname, "r");
  if (file == NULL) {printf("File %s doesn't exist\n", fname); exit(1); }
  printf("Opened file %s\n", fname);
  fseek(file, 0L, SEEK_END);
  numbytes = ftell(file);
  printf("Size is %lu\n", numbytes);

  // copy file into memory
  fseek(file, 0L, SEEK_SET);
  buf = (uint8_t*)calloc(numbytes, sizeof(uint8_t));
  if(buf == NULL) exit(1);
  if (fread(buf, 1, numbytes, file) != numbytes)
  {
    printf("Unable to read all bytes from file %s\n", fname);
    exit(EXIT_FAILURE);
  }
  fclose(file);

  // calculate the padding
  padding = numbytes % AES_BLOCK_SIZE;
  numbytes += padding;
  printf("Padding file with %d bytes for a new size of %lu\n", padding, numbytes);

  // generate key
  for (i = 0; i < sizeof(key);i++) key[i] = i;

  // this is to force nvcc to put the gpu initialization here
  GPU_init<<<1, 1>>>();

  // encryption
  start = clock();
  for (int k = 0; k < MEASUREMENT_TIMES; k++) {
    encryptdemo(key, buf, numbytes);
  }
  end = clock();
  printf("time used:%f\n",  (double)(end - start) / CLOCKS_PER_SEC /MEASUREMENT_TIMES);
  printf("GPU encryption throughput: %f bytes/second\n",  (double)(numbytes) / ((double)(end - start) / CLOCKS_PER_SEC / MEASUREMENT_TIMES));


  // write into file
  file = fopen("cipher.txt", "w");
  fwrite(buf, 1, numbytes, file);
  fclose(file);

  // decryption
  start = clock();
  for (int k = 0; k < MEASUREMENT_TIMES; k++) {
    decryptdemo(key, buf, numbytes);
  }
  end = clock();
  printf("time used:%f\n",  (double)(end - start) / CLOCKS_PER_SEC / MEASUREMENT_TIMES);
  printf("GPU encryption throughput: %f bytes/second\n",  (double)(numbytes) / ((double)(end - start) / CLOCKS_PER_SEC / MEASUREMENT_TIMES));


  // write into file
  file = fopen("output.txt", "w");
  fwrite(buf, 1, numbytes - padding, file);
  fclose(file);

  free(buf);
  return EXIT_SUCCESS;
}
