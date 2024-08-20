/**
 * @file aes.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <openssl/evp.h>
#include <openssl/err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <getopt.h>
#define MEASUREMENT_TIMES (1 << 8)
#define AES_BLOCK_SIZE 16 
#define AES_KEY_BUFFER_SIZE 32


// Params 
typedef struct Params
{
    uint64_t inputSize;
    char *keyFile;
    char *inputFile;
    char *cipherFile;
    char *outputFile;
    bool shouldVerify;
} Params;

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

void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

int encrypt(unsigned char *plaintext, int plaintext_len,
            unsigned char *key,
            unsigned char *ciphertext) {
    EVP_CIPHER_CTX *ctx;
    int len;
    int ciphertext_len;

    /* Create and initialise the context */
    if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();

    /* Initialise the encryption operation. */
    if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, NULL))
        handleErrors();

    /* Provide the message to be encrypted, and obtain the encrypted output. */
    if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
        handleErrors();
    ciphertext_len = len;

    /* Finalise the encryption. */
    if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) handleErrors();
    ciphertext_len += len;

    /* Clean up */
    EVP_CIPHER_CTX_free(ctx);

    return ciphertext_len;
}

int decrypt(unsigned char *ciphertext, int ciphertext_len,
            unsigned char *key,
            unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx;
    int len;
    int plaintext_len;
    int ret;

    /* Create and initialise the context */
    if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();

    /* Initialise the decryption operation. */
    if(!EVP_DecryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, NULL))
        handleErrors();

    /* Provide the message to be decrypted, and obtain the plaintext output. */
    if(1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len))
        handleErrors();
    plaintext_len = len;

    /* Finalise the decryption. */
    ret = EVP_DecryptFinal_ex(ctx, plaintext + len, &len);

    /* Clean up */
    EVP_CIPHER_CTX_free(ctx);

    if(ret > 0) {
        plaintext_len += len;
        return plaintext_len;
    } else {
        return -1;
    }
}
// Function to compare two files
int compare_files(const char *file1, const char *file2);


int main(int argc, char *argv[]) {

    struct Params params = getInputParams(argc, argv);

    FILE *file;
    uint8_t *plaintext;
    uint8_t *ciphertext;
    uint8_t *decryptedtext;
    unsigned long numbytes;
    int padding;
    uint8_t key[AES_KEY_BUFFER_SIZE]; // Encryption/Decryption key.

    if (params.keyFile == NULL) {
        printf("INFO: Key file is not specifed. Random key will be used.\n");
        for (unsigned int i = 0; i < AES_KEY_BUFFER_SIZE; ++i) {
            key[i] = rand() & 0xff;
        }
    } else {
        // Open and read the key file.
        file = fopen(params.keyFile, "r");
        if (file == NULL) {
            printf("ERROR: Error opening key file %s\n", params.keyFile);
            return EXIT_FAILURE;
        }
        if (fread(key, 1, AES_KEY_BUFFER_SIZE, file) != AES_KEY_BUFFER_SIZE) {
          printf("ERROR: The key length in %s is not %d characters\n", params.keyFile, AES_KEY_BUFFER_SIZE);
          fclose(file);
          return EXIT_FAILURE;
        } 
        // Verify that there are no extra characters.
        char extra;
        if (fread(&extra, 1, 1, file) != 0) {
            printf("ERROR: The key length in %s is more than %d characters\n", params.keyFile, AES_KEY_BUFFER_SIZE);
            fclose(file);
            return EXIT_FAILURE;
        }
        fclose(file);
    }

    // Allocate memory for the file content.
    numbytes = params.inputSize;
    plaintext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    if (plaintext == NULL) {
        printf("ERROR: Memory allocation error\n");
        return EXIT_FAILURE;
    }
    ciphertext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    if (ciphertext == NULL) {
        printf("ERROR: Memory allocation error\n");
        return EXIT_FAILURE;
    }
    decryptedtext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    if (decryptedtext == NULL) {
        printf("ERROR: Memory allocation error\n");
        return EXIT_FAILURE;
    }
 
    // Open and read the input file.
    if (params.inputFile == NULL) {
        printf("INFO: Input file is not specifed. Random input will be used.\n");
        for (unsigned int i = 0; i < params.inputSize; ++i) {
            plaintext[i] = rand() & 0xff;
        }
    }
    else {
        file = fopen(params.inputFile, "r");
        if (file == NULL) {
            printf("ERROR: Error opening input file %s\n", params.inputFile);
            free(plaintext);
            free(ciphertext);
            free(decryptedtext);
            return EXIT_FAILURE;
        } 
        fseek(file, 0L, SEEK_END);
        numbytes = ftell(file);
        fseek(file, 0L, SEEK_SET);
        if (fread(plaintext, 1, numbytes, file) != numbytes) {
          printf("ERROR: Unable to read all bytes from file %s\n", params.inputFile);
          fclose(file);
          free(plaintext);
          free(ciphertext);
          free(decryptedtext);
          return EXIT_FAILURE;
        }
        fclose(file);
    }

    unsigned int ciphertextLen; 
    // Start encrypt in CPU
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < MEASUREMENT_TIMES; k++) {
        ciphertextLen = encrypt(plaintext, numbytes, key, ciphertext);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start) / MEASUREMENT_TIMES;
    std::cout << "INFO: Encryption Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

    // Write the ciphertext to file
    file = fopen(params.cipherFile, "w");
    if (file == NULL) {
        printf("ERROR: Error opening cipher file %s\n", params.cipherFile);
        free(plaintext);
        free(ciphertext);
        free(decryptedtext);
        return EXIT_FAILURE;
    } 
    fwrite(ciphertext, 1, ciphertextLen, file);
    fclose(file);

    // Start decrypt in CPU
    unsigned int decryptedtextLen;
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < MEASUREMENT_TIMES; k++) {
        decryptedtextLen = decrypt(ciphertext, ciphertextLen, key, decryptedtext);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start) / MEASUREMENT_TIMES;
    std::cout << "INFO: Decryption Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

    // Write to output file
    file = fopen(params.outputFile, "w");
    if (file == NULL) {
          printf("ERROR: Error opening output file %s\n", params.outputFile);
          free(plaintext);
          free(ciphertext);
          free(decryptedtext);
          return EXIT_FAILURE;
    } 
    fwrite(decryptedtext, 1, numbytes, file);
    fclose(file);


    // Compare input and output files
    if (params.shouldVerify) { 
        if (compare_files(params.inputFile, params.outputFile) == 0) {
            printf("INFO: The input file and the output file are the same.\n");
        } else {
            printf("ERROR: The input file and the output file are different.\n");
        }
    }

    free(plaintext);
    free(ciphertext);
    free(decryptedtext);
    return EXIT_SUCCESS;
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
