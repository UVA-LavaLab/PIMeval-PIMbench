#include <stdio.h>
#include <time.h> // for measuring CPU time
#include <stdlib.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>

#include "util.h"
#include "libpimeval.h"

typedef struct Params
{
  uint64_t numElements;
  char *configFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./round-trip.out [options]"
          "\n"
          "\n    -n    number of elements (default = 100)"
          "\n    -c    dramsim config file"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.numElements = 100;
  p.configFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "hn:c:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.numElements = atoi(optarg);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }

  assert(p.numElements > 0 && "Invalid number of elements");
  return p;
}

void transposeTest_sameMembers(uint64_t numElements) {
    struct Test {
        uint64_t field0;
        uint64_t field1;
        uint64_t field2;
        uint64_t field3;
    };
    Test* in = new Test[numElements];

    for (uint64_t i = 0; i < numElements; i++) {
        in[i].field0 = i * 10;
        in[i].field1 = i * 10 + 1;
        in[i].field2 = i * 10 + 2;
        in[i].field3 = i * 10 + 3;
    }

    PimObjId pimTest[4];

    pimTest[0] = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT64);
    assert(pimTest[0] != -1);


    for (int i = 1; i < 4; i++) {
        pimTest[i] = pimAllocAssociated(pimTest[0], PIM_UINT64);
        assert(pimTest[i] != -1);
    }

    PimStatus status;

    status = pimCopyHostToDeviceTranspose(in, pimTest[0], &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[1], &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[2], &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[3], &Test::field3, 0, numElements);
    assert(status == PIM_OK);

    Test* out = new Test[numElements];
    memset(out, 0, numElements * sizeof(Test));

    status = pimCopyDeviceToHostTranspose(pimTest[0], out, &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[1], out, &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[2], out, &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[3], out, &Test::field3, 0, numElements);
    assert(status == PIM_OK);

    printf("IN vs OUT\n");

    bool passed = true;
    for (uint64_t i = 0; i < numElements; i++) {
        printf("i = %lu\n", i);
        printf("field0: %lu    %lu\n", in[i].field0, out[i].field0);
        printf("field1: %lu    %lu\n", in[i].field1, out[i].field1);
        printf("field2: %lu    %lu\n", in[i].field2, out[i].field2);
        printf("field3: %lu    %lu\n\n", in[i].field3, out[i].field3);
        if (in[i].field0 != out[i].field0 ||
            in[i].field1 != out[i].field1 ||
            in[i].field2 != out[i].field2 ||
            in[i].field3 != out[i].field3) {
                passed = false;
                printf("Mismatch at index %lu\n", i);
                break;
        }
    }

    if (passed) {
        printf("Passed\n");
    }

    for (int i = 0; i < 4; i++) {
        pimFree(pimTest[i]);
    }
    
}


void transposeTest_diffMembers(uint64_t numElements) {
    struct Test {
        uint64_t field0;
        uint32_t field1;
        int64_t field2;
        float field3;
        int32_t field4;
    };

    Test* in = new Test[numElements];

    for (uint64_t i = 0; i < numElements; i++) {
        in[i].field0 = i * 10;
        in[i].field1 = i * 10 + 1;
        in[i].field2 = i * 10 + 2;
        in[i].field3 = i * 10 + 3;
        in[i].field4 = i * 10 + 4;
    }

    PimObjId pimTest[5];

    pimTest[0] = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT64);
    assert(pimTest[0] != -1);
    pimTest[1] = pimAllocAssociated(pimTest[0], PIM_UINT32);
    assert(pimTest[1] != -1);
    pimTest[2] = pimAllocAssociated(pimTest[0], PIM_INT64);
    assert(pimTest[2] != -1);
    pimTest[3] = pimAllocAssociated(pimTest[0], PIM_FP32);
    assert(pimTest[3] != -1);
    pimTest[4] = pimAllocAssociated(pimTest[0], PIM_INT32);
    assert(pimTest[4] != -1);

    PimStatus status;

    status = pimCopyHostToDeviceTranspose(in, pimTest[0], &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[1], &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[2], &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[3], &Test::field3, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[4], &Test::field4, 0, numElements);
    assert(status == PIM_OK);

    Test* out = new Test[numElements];
    memset(out, 0, numElements * sizeof(Test));

    status = pimCopyDeviceToHostTranspose(pimTest[0], out, &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[1], out, &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[2], out, &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[3], out, &Test::field3, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[4], out, &Test::field4, 0, numElements);
    assert(status == PIM_OK);

    printf("IN vs OUT\n");

    bool passed = true;
    for (uint64_t i = 0; i < numElements; i++) {
        printf("i = %lu\n", i);
        printf("field0: %lu    %lu\n", in[i].field0, out[i].field0);
        printf("field1: %u    %u\n", in[i].field1, out[i].field1);
        printf("field2: %ld    %ld\n", in[i].field2, out[i].field2);
        printf("field3: %f    %f\n\n", in[i].field3, out[i].field3);
        printf("field4: %u    %u\n\n", in[i].field4, out[i].field4);
        if (in[i].field0 != out[i].field0 ||
            in[i].field1 != out[i].field1 ||
            in[i].field2 != out[i].field2 ||
            in[i].field3 != out[i].field3 ||
            in[i].field4 != out[i].field4) {
                passed = false;
                printf("Mismatch at index %lu\n", i);
                break;
        }
    }

    if (passed) {
        printf("Passed\n");
    }

    for (int i = 0; i < 5; i++) {
        pimFree(pimTest[i]);
    }
    
}

void transposeTest_padding(uint64_t numElements) {
    struct Test {
        char field0;
        float field1;
        char field2;
        int field3;
    };
    Test* in = new Test[numElements];

    for (uint64_t i = 0; i < numElements; i++) {
        in[i].field0 = i;
        in[i].field1 = i * 10 + 1;
        in[i].field2 = i * 10 + 2;
        in[i].field3 = i * 10 + 3;
    }

    PimObjId pimTest[4];

    pimTest[0] = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT8);
    assert(pimTest[0] != -1);
    pimTest[1] = pimAllocAssociated(pimTest[0], PIM_FP32);
    assert(pimTest[1] != -1);
    pimTest[2] = pimAllocAssociated(pimTest[0], PIM_INT8);
    assert(pimTest[2] != -1);
    pimTest[3] = pimAllocAssociated(pimTest[0], PIM_INT32);
    assert(pimTest[3] != -1);

    PimStatus status;

    status = pimCopyHostToDeviceTranspose(in, pimTest[0], &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[1], &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[2], &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[3], &Test::field3, 0, numElements);
    assert(status == PIM_OK);

    Test* out = new Test[numElements];
    memset(out, 0, numElements * sizeof(Test));

    status = pimCopyDeviceToHostTranspose(pimTest[0], out, &Test::field0, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[1], out, &Test::field1, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[2], out, &Test::field2, 0, numElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[3], out, &Test::field3, 0, numElements);
    assert(status == PIM_OK);

    printf("IN vs OUT\n");

    bool passed = true;
    for (uint64_t i = 0; i < numElements; i++) {
        printf("i = %lu\n", i);
        printf("field0: %u    %u\n", in[i].field0, out[i].field0);
        printf("field1: %f    %f\n", in[i].field1, out[i].field1);
        printf("field2: %u    %u\n", in[i].field2, out[i].field2);
        printf("field3: %u    %u\n\n", in[i].field3, out[i].field3);
        if (in[i].field0 != out[i].field0 ||
            in[i].field1 != out[i].field1 ||
            in[i].field2 != out[i].field2 ||
            in[i].field3 != out[i].field3) {
                passed = false;
                printf("Mismatch at index %lu\n", i);
                break;
        }
    }

    if (passed) {
        printf("Passed\n");
    }

    for (int i = 0; i < 4; i++) {
        pimFree(pimTest[i]);
    }
    
}

void transposeTest_partialRange(uint64_t numElements) {
    struct Test {
        char field0;
        float field1;
        char field2;
        int field3;
    };
    Test* in = new Test[numElements];

    for (uint64_t i = 0; i < numElements; i++) {
        in[i].field0 = i;
        in[i].field1 = i * 10 + 1;
        in[i].field2 = i * 10 + 2;
        in[i].field3 = i * 10 + 3;
    }

    PimObjId pimTest[4];

    pimTest[0] = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT8);
    assert(pimTest[0] != -1);
    pimTest[1] = pimAllocAssociated(pimTest[0], PIM_FP32);
    assert(pimTest[1] != -1);
    pimTest[2] = pimAllocAssociated(pimTest[0], PIM_INT8);
    assert(pimTest[2] != -1);
    pimTest[3] = pimAllocAssociated(pimTest[0], PIM_INT32);
    assert(pimTest[3] != -1);

    PimStatus status;

    uint64_t start = numElements / 2;
    uint64_t end = (numElements / 4) * 3;

    status = pimCopyHostToDeviceTranspose(in, pimTest[0], &Test::field0, start, end);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[1], &Test::field1, start, end);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[2], &Test::field2, start, end);
    assert(status == PIM_OK);
    status = pimCopyHostToDeviceTranspose(in, pimTest[3], &Test::field3, start, end);
    assert(status == PIM_OK);

    Test* out = new Test[numElements];
    memset(out, 0, numElements * sizeof(Test));

    status = pimCopyDeviceToHostTranspose(pimTest[0], out, &Test::field0, start, end);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[1], out, &Test::field1, start, end);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[2], out, &Test::field2, start, end);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHostTranspose(pimTest[3], out, &Test::field3, start, end);
    assert(status == PIM_OK);

    printf("IN vs OUT\n");

    bool passed = true;
    for (uint64_t i = start; i < end; i++) {
        printf("i = %lu\n", i);
        printf("field0: %u    %u\n", in[i].field0, out[i].field0);
        printf("field1: %f    %f\n", in[i].field1, out[i].field1);
        printf("field2: %u    %u\n", in[i].field2, out[i].field2);
        printf("field3: %u    %u\n\n", in[i].field3, out[i].field3);
        if (in[i].field0 != out[i].field0 ||
            in[i].field1 != out[i].field1 ||
            in[i].field2 != out[i].field2 ||
            in[i].field3 != out[i].field3) {
                passed = false;
                printf("Mismatch at index %lu\n", i);
                break;
        }
    }

    if (passed) {
        printf("Passed\n");
    }

    for (int i = 0; i < 4; i++) {
        pimFree(pimTest[i]);
    }
    
}

int main(int argc, char *argv[])
{
  struct Params p = input_params(argc, argv);
  if (!createDevice(p.configFile))
  {
    return 1;
  }

  printf("------ TEST 1: Struct members of same type -------\n");
  transposeTest_sameMembers(p.numElements);

  printf("------ TEST 2: Struct members of different types -------\n");
  transposeTest_diffMembers(p.numElements);

  printf("------ TEST 3: Worst case struct padding -------\n");
  transposeTest_padding(p.numElements);

  printf("------ TEST 4: Partial range -------\n");
  transposeTest_partialRange(p.numElements);

  pimShowStats();
}
