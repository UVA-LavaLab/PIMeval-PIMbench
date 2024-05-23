// Image Downsampling implementation on bitSIMD
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "../util.h"
#include "libpimsim.h"

using namespace std;

typedef struct Params
{
  char *configFile;
  char *inputFile;
  bool shouldVerify;
  char *outputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./image_downsampling [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    input image file of BMP type (default=\"Dataset/input_1.bmp\")"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n    -o    output file for downsampled image (default=no output)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = "Dataset/input_1.bmp";
  p.shouldVerify = false;
  p.outputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:i:v:o:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'o':
      p.outputFile = optarg;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int saved_stdout_fd = -1;
FILE* saved_stdout = nullptr;

void disable_stdout()
{
  if (saved_stdout_fd != -1) {
    return;
  }

  fflush(stdout);

  saved_stdout_fd = dup(fileno(stdout));
  saved_stdout = fdopen(saved_stdout_fd, "w");

  freopen("/dev/null", "w", stdout);
}

void enable_stdout()
{
  if (saved_stdout_fd == -1) {
    return;
  }

  fflush(stdout);

  dup2(saved_stdout_fd, fileno(stdout));
  stdout = saved_stdout;

  saved_stdout_fd = -1;
  saved_stdout = nullptr;
}

struct NewImgWrapper {
  std::vector<uint8_t> new_img;
  int new_height;
  int new_width;
  int old_pixels_size;
  int scanline_size;
  int data_offset;
  int new_scanline_size;
  int new_data_offset;
  int new_pixel_data_width;
};

NewImgWrapper createNewImage(std::vector<uint8_t> img, bool print_size=false)
{
  NewImgWrapper res;

  res.data_offset = *((int*)(img.data() + 0xA));

  int img_width = *((int*)(img.data() + 0x12));
  int img_height = *((int*)(img.data() + 0x16));

  if(print_size) {
    printf("Input Image: %dx%d\n", img_width, img_height);
  }

  int x_pixels_per_m = *((int*)(img.data() + 0x26));
  int y_pixels_per_m = *((int*)(img.data() + 0x2A));

  res.scanline_size = 3 * img_width;
  int scanline_size_mod = res.scanline_size % 4;
  if (scanline_size_mod) {
    res.scanline_size = res.scanline_size - scanline_size_mod + 4;
  }

  res.old_pixels_size = img_height * res.scanline_size;

  res.new_pixel_data_width = 3 * (img_width >> 1);
  res.new_scanline_size = res.new_pixel_data_width;
  int new_scanline_size_mod = res.new_scanline_size % 4;
  if (new_scanline_size_mod) {
    res.new_scanline_size = res.new_scanline_size - new_scanline_size_mod + 4;
  }

  res.new_width = img_width >> 1;
  res.new_height = img_height >> 1;
  res.new_data_offset = 0x36;
  int new_img_size = res.new_data_offset + res.new_height * res.new_scanline_size;
  res.new_img.resize(new_img_size);

  //    Header
  res.new_img[0] = 'B';
  res.new_img[1] = 'M';
  *((int*)(res.new_img.data() + 2)) = new_img_size;
  *((int*)(res.new_img.data() + 6)) = 0;
  *((int*)(res.new_img.data() + 0xA)) = res.new_data_offset;

  //    InfoHeader
  *((int*)(res.new_img.data() + 0xE)) = 40;
  *((int*)(res.new_img.data() + 0x12)) = res.new_width;
  *((int*)(res.new_img.data() + 0x16)) = res.new_height;
  *((int16_t*)(res.new_img.data() + 0x1A)) = 1;
  *((int16_t*)(res.new_img.data() + 0x1C)) = 24;
  *((int*)(res.new_img.data() + 0x1E)) = 0;
  *((int*)(res.new_img.data() + 0x22)) = 0;
  *((int*)(res.new_img.data() + 0x26)) = x_pixels_per_m;
  *((int*)(res.new_img.data() + 0x2A)) = y_pixels_per_m;
  *((int*)(res.new_img.data() + 0x2E)) = 0;
  *((int*)(res.new_img.data() + 0x32)) = 0;

  return res;
}

void fullAdder(PimObjId A_obj, uint8_t A_offset, PimObjId B_obj,
  uint8_t B_offset, PimObjId pix_out_obj, uint8_t output_offset,
  PimRowReg Cin, PimRowReg Cout, bool output_enable,
  bool cout_enable)
{
  pimOpReadRowToSa(B_obj, B_offset);
  pimOpMove(B_obj, PIM_RREG_SA, PIM_RREG_R1);  // Read bit of input1 into r1
  pimOpReadRowToSa(A_obj, A_offset);
  // A=SA, B=R1, Cin = R2
  // Cout = R3
  if (cout_enable) {
    pimOpMaj(A_obj, PIM_RREG_SA, PIM_RREG_R1, Cin, Cout);
  }
  if (output_enable) {
    pimOpXor(A_obj, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpXor(A_obj, PIM_RREG_SA, Cin, PIM_RREG_SA);
    pimOpWriteSaToRow(pix_out_obj, output_offset);
  }
  // Calling with alternating Cin/Couts allows me to remove this move
  // pimOpMove(pix_in_obj, PIM_RREG_R3, PIM_RREG_R2);
}

size_t pimAverageRows(uint8_t* output, uint8_t* in_row_1, size_t sz_1, uint8_t* in_row_2, size_t sz_2)
{
  assert(sz_1 == sz_2);
  assert(0 == sz_1 % 6);

  int width_pix = sz_1 / 3;

  PimObjId pix_in_1_obj = pimAlloc(PIM_ALLOC_V1, width_pix >> 1, 48, PIM_INT64);
  assert(-1 != pix_in_1_obj);

  PimObjId pix_in_2_obj = pimAllocAssociated(PIM_ALLOC_V1, width_pix >> 1, 48, pix_in_1_obj, PIM_INT64);
  assert(-1 != pix_in_2_obj);

  PimObjId pix_out_obj = pimAllocAssociated(PIM_ALLOC_V1, width_pix >> 1, 24, pix_in_1_obj, PIM_INT32);
  assert(-1 != pix_out_obj);

  PimStatus row_1_copy_status = pimCopyHostToDevice(PIM_COPY_V, in_row_1, pix_in_1_obj);
  assert(PIM_OK == row_1_copy_status);

  PimStatus row_2_copy_status = pimCopyHostToDevice(PIM_COPY_V, in_row_2, pix_in_2_obj);
  assert(PIM_OK == row_2_copy_status);

  for (int j = 0; j < 3; ++j) {
    uint8_t input_offset_1 = 8 * j;
    uint8_t input_offset_2 = 24 + input_offset_1;
    uint8_t input_offset_3 = input_offset_1;
    uint8_t input_offset_4 = input_offset_2;
    uint8_t output_offset = input_offset_1;
    PimStatus s = pimOpSet(pix_in_1_obj, PIM_RREG_R2, 0);  // Set Carry in = 0
    assert(PIM_OK == s);
    pimOpSet(pix_in_1_obj, PIM_RREG_R4, 0);
    for (int i = 0; i < 8; i += 2) {
      // Adder logic to add two bytes and produce a shifted result
      // Optimized to minimize pim ops
      fullAdder(pix_in_1_obj, input_offset_1 + i, pix_in_1_obj, input_offset_2 + i, pix_in_1_obj, output_offset + i, PIM_RREG_R2, PIM_RREG_R3, true, true);
      fullAdder(pix_in_2_obj, input_offset_3 + i, pix_in_2_obj, input_offset_4 + i, pix_in_2_obj, output_offset + i, PIM_RREG_R4, PIM_RREG_R5, true, true);

      fullAdder(pix_in_1_obj, input_offset_1 + i + 1, pix_in_1_obj, input_offset_2 + i + 1, pix_in_1_obj, output_offset + i + 1, PIM_RREG_R3, PIM_RREG_R2, true, true);
      fullAdder(pix_in_2_obj, input_offset_3 + i + 1, pix_in_2_obj, input_offset_4 + i + 1, pix_in_2_obj, output_offset + i + 1, PIM_RREG_R5, PIM_RREG_R4, true, true);
    }

    // SA - used by adder
    // R1 - used by adder
    // R2 - Cout of first block
    // R3 - Cin of second block
    // R4 - Cout of first block
    // R5 - Interemediate of second block

    pimOpSet(pix_in_1_obj, PIM_RREG_R3, 0);  // Set Carry in = 0
    for (int i = 0; i < 8; i += 2) {
      // Adder logic to add two bytes and produce a shifted result
      // Optimized to minimize pim ops
      fullAdder(pix_in_1_obj, input_offset_1 + i, pix_in_2_obj, input_offset_3 + i, pix_out_obj, output_offset + i - 2, PIM_RREG_R3, PIM_RREG_R5, i > 0, true);
      fullAdder(pix_in_1_obj, input_offset_1 + i + 1, pix_in_2_obj, input_offset_3 + i + 1, pix_out_obj, output_offset + i - 1, PIM_RREG_R5, PIM_RREG_R3, i > 0, true);
    }
    pimOpXor(pix_out_obj, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_SA);
    pimOpXor(pix_out_obj, PIM_RREG_SA, PIM_RREG_R4, PIM_RREG_SA);
    pimOpWriteSaToRow(pix_out_obj, output_offset + 6);

    pimOpMaj(pix_out_obj, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R4, PIM_RREG_SA);
    pimOpWriteSaToRow(pix_out_obj, output_offset + 7);
  }

  PimStatus output_copy_status = pimCopyDeviceToHost(PIM_COPY_V, pix_out_obj, (void*)output);
  assert(PIM_OK == output_copy_status);

  pimFree(pix_in_1_obj);
  pimFree(pix_in_2_obj);
  pimFree(pix_out_obj);

  return sz_1 >> 1;
}

std::vector<uint8_t> avg_pim(std::vector<uint8_t>& img)
{
  NewImgWrapper avg_out = createNewImage(img, true);
  uint8_t* pixels_out_avg = (uint8_t*)avg_out.new_img.data() + avg_out.new_data_offset;
  uint8_t* pixels_in = (uint8_t*)img.data() + avg_out.data_offset;

  uint8_t* pixels_out_avg_it = pixels_out_avg;
  uint8_t* pixels_in_it = pixels_in;

  for (int y = 0; y < avg_out.new_height; ++y) {
    disable_stdout();
    pimAverageRows(pixels_out_avg_it, pixels_in_it, 2 * avg_out.new_pixel_data_width, pixels_in_it + avg_out.scanline_size, 2 * avg_out.new_pixel_data_width);
    enable_stdout();

    // Set 0 padding to nearest 4 byte boundary as required by BMP standard [1]
    for (int x = avg_out.new_pixel_data_width; x < avg_out.new_scanline_size; ++x) {
      pixels_out_avg[avg_out.new_scanline_size * y + x] = 0;
    }
    pixels_in_it += 2 * avg_out.scanline_size;
    pixels_out_avg_it += avg_out.new_scanline_size;
  }

  return avg_out.new_img;
}

struct Pixel {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
};

inline Pixel* get_pixel(const char* pixels, int scanline_size, int x, int y)
{
  return (Pixel*)(pixels + scanline_size * y + 3 * x);
}

inline void set_pixel(const char* pixels, Pixel* new_pixel, int scanline_size, int x, int y)
{
  auto* old_pix = (Pixel*)(pixels + scanline_size * y + 3 * x);
  *old_pix = *new_pixel;
}

std::vector<uint8_t> avg_cpu(std::vector<uint8_t> img)
{
  //    Averaging Kernel
  NewImgWrapper avg_out = createNewImage(img);
  char* pixels_out_averaged = (char*)avg_out.new_img.data() + avg_out.new_data_offset;
  char* pixels_in = (char*)img.data() + avg_out.data_offset;

  for (int y = 0; y < avg_out.new_height; ++y) {
    for (int x = 0; x < avg_out.new_width; ++x) {  // 4 per get pixel
      Pixel curr_pix1 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x, 2 * y);  // 4 + 2
      Pixel curr_pix2 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y);  // 4 + 3
      Pixel curr_pix3 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x, 2 * y + 1);  // 4 + 3
      Pixel curr_pix4 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y + 1);  // 4 + 4

      Pixel new_pix;
      new_pix.red = (((uint16_t)curr_pix1.red) + ((uint16_t)curr_pix2.red) + ((uint16_t)curr_pix3.red) + ((uint16_t)curr_pix4.red)) >> 2;
      new_pix.blue = (((uint16_t)curr_pix1.blue) + ((uint16_t)curr_pix2.blue) + ((uint16_t)curr_pix3.blue) + ((uint16_t)curr_pix4.blue)) >> 2;
      new_pix.green = (((uint16_t)curr_pix1.green) + ((uint16_t)curr_pix2.green) + ((uint16_t)curr_pix3.green) + ((uint16_t)curr_pix4.green)) >> 2;

      set_pixel(pixels_out_averaged, &new_pix, avg_out.new_scanline_size, x, y);
    }
    // Set 0 padding to nearest 4 byte boundary [1]
    for (int x = avg_out.new_pixel_data_width; x < avg_out.new_scanline_size; ++x) {
      pixels_out_averaged[avg_out.new_scanline_size * y + x] = 0;
    }
  }
  return avg_out.new_img;
}

vector<uint8_t> read_file_bytes(char* filename)
{
  ifstream img_file(filename, std::ios::ate | std::ios::binary);
  streamsize img_size = img_file.tellg();
  img_file.seekg(0, std::ios::beg);

  vector<uint8_t> img_buffer(img_size);
  if (!img_file.read((char*)img_buffer.data(), img_size)) {
    throw runtime_error("Error reading image file!");
  }

  return img_buffer;
}

void write_img(vector<uint8_t>& img, std::string filename) {
    auto outfile = std::fstream(filename, std::ios::out | std::ios::binary);
    outfile.write((char*) img.data(), img.size());
    outfile.close();
}

bool check_image(std::vector<uint8_t>& img) {
  if (img[0] != 'B' || img[1] != 'M') {
    cout << "Not a BMP file!\n";
    return false;
  }

  int compression = *((int*)(img.data() + 0x1E));
  if (compression) {
    cout << "Error, compressed bmp files not supported\n";
    return false;
  }

  int16_t bits_per_pixel = *((int16_t*)(img.data() + 0x1C));
  if (bits_per_pixel != 24) {
    cout << "Only 24 bits per pixel currently supported\n";
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "PIM test: Image Downsampling" << std::endl;

  std::vector<uint8_t> img = read_file_bytes(params.inputFile);

  if(!createDevice(params.configFile)) {
    return 1;
  }

  if(!check_image(img)) {
    return 1;
  }

  vector<uint8_t> pim_averaged = avg_pim(img);

  if(params.shouldVerify) {
    vector<uint8_t> cpu_averaged = avg_cpu(img);

    if (cpu_averaged.size() != pim_averaged.size()) {
      cout << "Average kernel fail, sizes do not match" << endl;
      return 1;
    }
    for (size_t i = 0; i < cpu_averaged.size(); ++i) {
      if (cpu_averaged[i] != pim_averaged[i]) {
        cout << "Average kernel mismatch at byte " << i << endl;
        return 1;
      }
    }

    cout << "PIM Result matches CPU result" << endl;
  }
  pimShowStats();
  if(params.outputFile != nullptr) {
    write_img(pim_averaged, params.outputFile);
  }
}

// [1] N. Liesch, The bmp file format. [Online]. Available:
// https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm.
