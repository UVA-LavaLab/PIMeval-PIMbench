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
        "\n    -k    key file containing two vectors (default=generates key with random numbers)"
        "\n    -i    input file containing two vectors (default=generates input with random numbers)"
        "\n    -c    cipher file containing two vectors (default=./cipher.txt)"
        "\n    -o    output file containing two vectors (default=./output.txt)"
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
            key[i] = rand() && 0xff;
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
    plaintext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    ciphertext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    decryptedtext = (uint8_t*)malloc((numbytes + 100) * sizeof(uint8_t));
    if (plaintext == NULL) {
        printf("ERROR: Memory allocation error\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Read the file into the buffer.
    if (!generateRandomInput && fread(plaintext, 1, numbytes, file) != numbytes) {
        printf("ERROR: Unable to read all bytes from file %s\n", params.inputFile);
        fclose(file);
        free(plaintext);
        return EXIT_FAILURE;
    }
    if (!generateRandomInput) { 
        fclose(file);
    } else {
        for (unsigned int i = 0; i < params.inputSize; ++i) {
            plaintext[i] = rand() && 0xff;
        }
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
    fwrite(ciphertext, 1, ciphertextLen, file);
    fclose(file);

    unsigned int decryptedtextLen;
    // Start decrypt in CPU
    start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < MEASUREMENT_TIMES; k++) {
        decryptedtextLen = decrypt(ciphertext, ciphertextLen, key, decryptedtext);

    }
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start) / MEASUREMENT_TIMES;
    std::cout << "INFO: Decryption Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

    // Write to output file
    file = fopen(params.outputFile, "w");
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
