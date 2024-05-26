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

void pimAverageRows(vector<uint32_t>& upper_left, vector<uint32_t>& upper_right, vector<uint32_t>& lower_left, vector<uint32_t>& lower_right, uint8_t* result)
{
  int sz = upper_left.size();

  PimObjId ul = pimAlloc(PIM_ALLOC_V1, sz, 32, PIM_INT32);
  assert(-1 != ul);

  PimObjId ur = pimAllocAssociated(PIM_ALLOC_V1, sz, 32, ul, PIM_INT32);
  assert(-1 != ur);

  PimObjId ll = pimAllocAssociated(PIM_ALLOC_V1, sz, 32, ul, PIM_INT32);
  assert(-1 != ll);

  PimObjId lr = pimAllocAssociated(PIM_ALLOC_V1, sz, 32, ul, PIM_INT32);
  assert(-1 != lr);

  PimObjId divisor_4 = pimAllocAssociated(PIM_ALLOC_V1, sz, 32, ul, PIM_INT32);
  assert(-1 != divisor_4);

  PimStatus ul_status = pimCopyHostToDevice(PIM_COPY_V, upper_left.data(), ul);
  assert(PIM_OK == ul_status);

  PimStatus ur_status = pimCopyHostToDevice(PIM_COPY_V, upper_right.data(), ur);
  assert(PIM_OK == ur_status);

  PimStatus ll_status = pimCopyHostToDevice(PIM_COPY_V, lower_left.data(), ll);
  assert(PIM_OK == ll_status);

  PimStatus lr_status = pimCopyHostToDevice(PIM_COPY_V, lower_right.data(), lr);
  assert(PIM_OK == lr_status);

  PimStatus divisor_4_status = pimBroadCast(PIM_COPY_V, divisor_4, 4);
  assert(PIM_OK == divisor_4_status);

  PimStatus upper_sum_status = pimAdd(ul, ur, ur);
  assert(PIM_OK == upper_sum_status);

  PimStatus lower_sum_status = pimAdd(ll, lr, lr);
  assert(PIM_OK == lower_sum_status);

  PimStatus result_sum_status = pimAdd(ur, lr, lr);
  assert(PIM_OK == result_sum_status);

  PimStatus lr_div_status = pimDiv(lr, divisor_4, lr);
  assert(PIM_OK == lr_div_status);

  vector<uint32_t> tmp;
  tmp.resize(sz);

  PimStatus result_copy_status = pimCopyDeviceToHost(PIM_COPY_V, lr, (void*)tmp.data());
  assert(PIM_OK == result_copy_status);

  for(int i=0; i<sz; ++i) {
    result[i] = (uint8_t) tmp[i];
  }

  pimFree(ul);
  pimFree(ur);
  pimFree(ll);
  pimFree(lr);
}

std::vector<uint8_t> avg_pim(std::vector<uint8_t>& img, int pim_rows)
{
  // TODO: pack multiple pairs of rows into each PIM iteration
  NewImgWrapper avg_out = createNewImage(img, true);
  uint8_t* pixels_out_avg = (uint8_t*)avg_out.new_img.data() + avg_out.new_data_offset;
  uint8_t* pixels_in = (uint8_t*)img.data() + avg_out.data_offset;

  uint8_t* pixels_out_avg_it = pixels_out_avg;
  uint8_t* pixels_in_it = pixels_in;

  // Seperate out horzontally adjacent pixels into vectors in CPU
  int sz = 3*avg_out.new_width;
  vector<uint32_t> upper_left;
  upper_left.reserve(pim_rows);
  vector<uint32_t> upper_right;
  upper_right.reserve(pim_rows);
  vector<uint32_t> lower_left;
  lower_left.reserve(pim_rows);
  vector<uint32_t> lower_right;
  lower_right.reserve(pim_rows);
  for (int y = 0; y < avg_out.new_height; ++y) {
    // upper_left.clear();
    // upper_right.clear();
    // lower_left.clear();
    // lower_right.clear();
    uint8_t* row2_it = pixels_in_it + avg_out.scanline_size;
    for(int x = 0; x < 6*avg_out.new_width; x += 6) {
      for(int i=0; i<3; ++i) {
        upper_left.push_back((uint32_t) pixels_in_it[x+i]);
        upper_right.push_back((uint32_t) pixels_in_it[x+i+3]);
        lower_left.push_back((uint32_t) row2_it[x+i]);
        lower_right.push_back((uint32_t) row2_it[x+3+i]);
        if(pim_rows == upper_left.size()) {
          // TODO: avg rows and reset and continue
          pimAverageRows(upper_left, upper_right, lower_left, lower_right, pixels_out_avg_it);
          pixels_out_avg_it += pim_rows;
          upper_left.clear();
          upper_right.clear();
          lower_left.clear();
          lower_right.clear();
        }
      }

      // upper_left.push_back((uint32_t) pixels_in_it[x+1]);
      // upper_left.push_back((uint32_t) pixels_in_it[x+2]);
      
      // upper_right.push_back((uint32_t) pixels_in_it[x+4]);
      // upper_right.push_back((uint32_t) pixels_in_it[x+5]);

      // lower_left.push_back((uint32_t) row2_it[x]);
      // lower_left.push_back((uint32_t) row2_it[x+1]);
      // lower_left.push_back((uint32_t) row2_it[x+2]);
      // lower_right.push_back((uint32_t) row2_it[x+3]);
      // lower_right.push_back((uint32_t) row2_it[x+4]);
      // lower_right.push_back((uint32_t) row2_it[x+5]);
      // if(upper_left.size() == pim_rows) {

      // }
    }
    // pimAverageRows(upper_left, upper_right, lower_left, lower_right, pixels_out_avg_it);

    // Set 0 padding to nearest 4 byte boundary as required by BMP standard [1]
    for(int x=0; x<(avg_out.new_scanline_size - avg_out.new_pixel_data_width); ++x) {
      upper_left.push_back(0);
      upper_right.push_back(0);
      lower_left.push_back(0);
      lower_right.push_back(0);
      if(pim_rows == upper_left.size()) {
        // TODO: avg rows and reset and continue
        pimAverageRows(upper_left, upper_right, lower_left, lower_right, pixels_out_avg_it);
        pixels_out_avg_it += pim_rows;
        upper_left.clear();
        upper_right.clear();
        lower_left.clear();
        lower_right.clear();
      }
    }
    // for (int x = avg_out.new_pixel_data_width; x < avg_out.new_scanline_size; ++x) {
    //   pixels_out_avg[avg_out.new_scanline_size * y + x] = 0;
    // }
    pixels_in_it += 2 * avg_out.scanline_size;
    // pixels_out_avg_it += avg_out.new_scanline_size;
    if(upper_left.size() != 0) {
      // TODO avg rows
      pimAverageRows(upper_left, upper_right, lower_left, lower_right, pixels_out_avg_it);
    }
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

  int pim_rows = 8192;

  if(!createDevice(params.configFile)) {
    return 1;
  }

  if(!check_image(img)) {
    return 1;
  }

  vector<uint8_t> pim_averaged = avg_pim(img, pim_rows);

  if(params.outputFile != nullptr) {
    write_img(pim_averaged, params.outputFile);
  }

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
}

// [1] N. Liesch, The bmp file format. [Online]. Available:
// https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm.
