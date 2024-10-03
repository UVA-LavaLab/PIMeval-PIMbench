// Image Downsampling implementation on bitSIMD
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h>

#include "../../util.h"
#include "libpimeval.h"

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
          "\nUsage:  ./image-downsampling.out [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    input image file of BMP type (default=\"input_1.bmp\")"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n    -o    output file for downsampled image (default=no output)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = (char*) "input_1.bmp";
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


// BMP file offsets constants [1], used to parse BMP file
constexpr int data_offset_location = 0xA;
constexpr int width_location = 0x12;
constexpr int height_location = 0x16;
constexpr int x_pixels_per_m_location = 0x26;
constexpr int y_pixels_per_m_location = 0x2A;
constexpr int size_location = 2;
constexpr int unused_location = 6;
constexpr int info_header_size_location = 0xE;
constexpr int planes_location = 0x1A;
constexpr int bit_per_pixel_location = 0x1C;
constexpr int compression_location = 0x1E;
constexpr int compressed_size_location = 0x22;
constexpr int used_colors_location = 0x2E;
constexpr int important_colors_location = 0x32;

constexpr int info_header_size = 40;
constexpr int num_planes = 1;
constexpr int bits_per_pixel = 24;
constexpr int min_data_offset = 0x36;
constexpr int color_channels_per_pixel = 3;
constexpr int bmp_scanline_padding_multiple = 4; // BMP file format specifies that scanlines are padded to nearest 4 byte boundary
constexpr int shift_amount = 2;

NewImgWrapper parseInputImageandSetupOutputImage(std::vector<uint8_t> img, bool print_size=false)
{
  // Parse input BMP file and create header for output BMP file
  // Parse BMP file [1]

  NewImgWrapper res;
  res.data_offset = *((int*)(img.data() + data_offset_location));

  int img_width = *((int*)(img.data() + width_location));
  int img_height = *((int*)(img.data() + height_location));

  if(print_size)
  {
    printf("Input Image: %dx%d\n", img_width, img_height);
  }

  int x_pixels_per_m = *((int*)(img.data() + x_pixels_per_m_location));
  int y_pixels_per_m = *((int*)(img.data() + y_pixels_per_m_location));

  // Scan line size is padded to nearest 4 byte boundary, below obtains the scan line size
  res.scanline_size = color_channels_per_pixel * img_width;
  int scanline_size_mod = res.scanline_size % bmp_scanline_padding_multiple;
  if (scanline_size_mod) {
    res.scanline_size = res.scanline_size - scanline_size_mod + bmp_scanline_padding_multiple;
  }

  res.old_pixels_size = img_height * res.scanline_size;

  res.new_pixel_data_width = color_channels_per_pixel * (img_width >> 1);

  // Scan line size is padded to nearest 4 byte boundary, below obtains the scan line size for the image being created
  res.new_scanline_size = res.new_pixel_data_width;
  int new_scanline_size_mod = res.new_scanline_size % bmp_scanline_padding_multiple;
  if (new_scanline_size_mod) {
    res.new_scanline_size = res.new_scanline_size - new_scanline_size_mod + bmp_scanline_padding_multiple;
  }

  res.new_width = img_width >> 1;
  res.new_height = img_height >> 1;
  res.new_data_offset = min_data_offset;
  int new_img_size = res.new_data_offset + res.new_height * res.new_scanline_size;
  res.new_img.resize(new_img_size);

  //    Header
  res.new_img[0] = 'B';
  res.new_img[1] = 'M';
  *((int*)(res.new_img.data() + size_location)) = new_img_size;
  *((int*)(res.new_img.data() + unused_location)) = 0;
  *((int*)(res.new_img.data() + data_offset_location)) = res.new_data_offset;

  //    InfoHeader
  *((int*)(res.new_img.data() + info_header_size_location)) = info_header_size;
  *((int*)(res.new_img.data() + width_location)) = res.new_width;
  *((int*)(res.new_img.data() + height_location)) = res.new_height;
  *((int16_t*)(res.new_img.data() + planes_location)) = num_planes;
  *((int16_t*)(res.new_img.data() + bit_per_pixel_location)) = bits_per_pixel;
  *((int*)(res.new_img.data() + compression_location)) = 0;
  *((int*)(res.new_img.data() + compressed_size_location)) = 0;
  *((int*)(res.new_img.data() + x_pixels_per_m_location)) = x_pixels_per_m;
  *((int*)(res.new_img.data() + y_pixels_per_m_location)) = y_pixels_per_m;
  *((int*)(res.new_img.data() + used_colors_location)) = 0;
  *((int*)(res.new_img.data() + important_colors_location)) = 0;

  return res;
}

void pimAverageRows(vector<uint8_t>& upper_left, vector<uint8_t>& upper_right, vector<uint8_t>& lower_left, vector<uint8_t>& lower_right, uint8_t* result)
{
  // Returns average of the four input vectors as a uint8_t array
  int sz = upper_left.size();

  PimObjId ul = pimAlloc(PIM_ALLOC_AUTO, sz, PIM_UINT8);
  assert(-1 != ul);

  PimObjId ur = pimAllocAssociated(ul, PIM_UINT8);
  assert(-1 != ur);

  PimObjId ll = pimAllocAssociated(ul, PIM_UINT8);
  assert(-1 != ll);

  PimObjId lr = pimAllocAssociated(ul, PIM_UINT8);
  assert(-1 != lr);

  PimStatus ul_status = pimCopyHostToDevice(upper_left.data(), ul);
  assert(PIM_OK == ul_status);

  PimStatus ur_status = pimCopyHostToDevice(upper_right.data(), ur);
  assert(PIM_OK == ur_status);

  PimStatus ll_status = pimCopyHostToDevice(lower_left.data(), ll);
  assert(PIM_OK == ll_status);

  PimStatus lr_status = pimCopyHostToDevice(lower_right.data(), lr);
  assert(PIM_OK == lr_status);

  PimStatus ul_right_shift_status = pimShiftBitsRight(ul, ul, shift_amount);
  assert(PIM_OK == ul_right_shift_status);

  PimStatus ur_right_shift_status = pimShiftBitsRight(ur, ur, shift_amount);
  assert(PIM_OK == ur_right_shift_status);

  PimStatus ll_right_shift_status = pimShiftBitsRight(ll, ll, shift_amount);
  assert(PIM_OK == ll_right_shift_status);

  PimStatus lr_right_shift_status = pimShiftBitsRight(lr, lr, shift_amount);
  assert(PIM_OK == lr_right_shift_status);

  PimStatus upper_sum_status = pimAdd(ul, ur, ur);
  assert(PIM_OK == upper_sum_status);

  PimStatus lower_sum_status = pimAdd(ll, lr, lr);
  assert(PIM_OK == lower_sum_status);

  PimStatus result_sum_status = pimAdd(ur, lr, lr);
  assert(PIM_OK == result_sum_status);

  PimStatus result_copy_status = pimCopyDeviceToHost(lr, (void*) result);
  assert(PIM_OK == result_copy_status);

  pimFree(ul);
  pimFree(ur);
  pimFree(ll);
  pimFree(lr);
}

std::vector<uint8_t> image_downsampling_pim(std::vector<uint8_t>& img)
{
  NewImgWrapper avg_out = parseInputImageandSetupOutputImage(img, true);
  size_t needed_elements = avg_out.new_height * avg_out.new_scanline_size;
  uint8_t* pixels_out_avg = (uint8_t*)avg_out.new_img.data() + avg_out.new_data_offset;
  uint8_t* pixels_in = (uint8_t*)img.data() + avg_out.data_offset;

  uint8_t* pixels_out_avg_it = pixels_out_avg;
  uint8_t* pixels_in_it = pixels_in;

  // Transform input bitmap to vectors of colors in CPU
  vector<uint8_t> upper_left;
  upper_left.reserve(needed_elements);
  vector<uint8_t> upper_right;
  upper_right.reserve(needed_elements);
  vector<uint8_t> lower_left;
  lower_left.reserve(needed_elements);
  vector<uint8_t> lower_right;
  lower_right.reserve(needed_elements);
  for (int y = 0; y < avg_out.new_height; ++y) {
    uint8_t* row2_it = pixels_in_it + avg_out.scanline_size;
    for(int x = 0; x < 2*color_channels_per_pixel*avg_out.new_width; x += 2*color_channels_per_pixel) {
      for(int i=0; i<color_channels_per_pixel; ++i) {
        upper_left.push_back(pixels_in_it[x+i]);
        upper_right.push_back(pixels_in_it[x+i+color_channels_per_pixel]);
        lower_left.push_back(row2_it[x+i]);
        lower_right.push_back(row2_it[x+color_channels_per_pixel+i]);
      }
    }

    // Set 0 padding to nearest 4 byte boundary as required by BMP standard [1]
    for(int x=0; x<(avg_out.new_scanline_size - avg_out.new_pixel_data_width); ++x) {
      upper_left.push_back(0);
      upper_right.push_back(0);
      lower_left.push_back(0);
      lower_right.push_back(0);
    }
    pixels_in_it += 2 * avg_out.scanline_size;
  }

  pimAverageRows(upper_left, upper_right, lower_left, lower_right, pixels_out_avg_it);

  return avg_out.new_img;
}

struct Pixel {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
};

inline Pixel* get_pixel(const char* pixels, int scanline_size, int x, int y)
{
  return (Pixel*)(pixels + scanline_size * y + color_channels_per_pixel * x);
}

inline void set_pixel(const char* pixels, Pixel* new_pixel, int scanline_size, int x, int y)
{
  auto* old_pix = (Pixel*)(pixels + scanline_size * y + color_channels_per_pixel * x);
  *old_pix = *new_pixel;
}

std::vector<uint8_t> image_downsampling_cpu(std::vector<uint8_t> img)
{
  //    Averaging Kernel
  NewImgWrapper avg_out = parseInputImageandSetupOutputImage(img);
  char* pixels_out_averaged = (char*)avg_out.new_img.data() + avg_out.new_data_offset;
  char* pixels_in = (char*)img.data() + avg_out.data_offset;

  for (int y = 0; y < avg_out.new_height; ++y) {
    for (int x = 0; x < avg_out.new_width; ++x) {  // 4 per get pixel
      Pixel curr_pix1 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x, 2 * y);  // 4 + 2
      Pixel curr_pix2 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y);  // 4 + 3
      Pixel curr_pix3 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x, 2 * y + 1);  // 4 + 3
      Pixel curr_pix4 = *get_pixel(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y + 1);  // 4 + 4

      Pixel new_pix;
      new_pix.red = (curr_pix1.red>>shift_amount) + (curr_pix2.red>>shift_amount) + (curr_pix3.red>>shift_amount) + (curr_pix4.red>>shift_amount);
      new_pix.blue = (curr_pix1.blue>>shift_amount) + (curr_pix2.blue>>shift_amount) + (curr_pix3.blue>>shift_amount) + (curr_pix4.blue>>shift_amount);
      new_pix.green = (curr_pix1.green>>shift_amount) + (curr_pix2.green>>shift_amount) + (curr_pix3.green>>shift_amount) + (curr_pix4.green>>shift_amount);

      set_pixel(pixels_out_averaged, &new_pix, avg_out.new_scanline_size, x, y);
    }
    // Set 0 padding to nearest 4 byte boundary [1]
    for (int x = avg_out.new_pixel_data_width; x < avg_out.new_scanline_size; ++x) {
      pixels_out_averaged[avg_out.new_scanline_size * y + x] = 0;
    }
  }
  return avg_out.new_img;
}

vector<uint8_t> read_file_bytes(const string& filename)
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

void write_img_to_file(vector<uint8_t>& img, std::string filename) {
    auto outfile = std::fstream(filename, std::ios::out | std::ios::binary);
    outfile.write((char*) img.data(), img.size());
    outfile.close();
}

bool check_valid_input_image(std::vector<uint8_t>& img) {
  // Verify that input image is of the correct type
  if (img[0] != 'B' || img[1] != 'M') {
    cout << "Not a BMP file!\n";
    return false;
  }

  int compression = *((int*)(img.data() + compression_location));
  if (compression) {
    cout << "Error, compressed bmp files not supported\n";
    return false;
  }

  int16_t actual_bits_per_pixel = *((int16_t*)(img.data() + bit_per_pixel_location));
  if (actual_bits_per_pixel != bits_per_pixel) {
    cout << "Only " << bits_per_pixel << " bits per pixel currently supported, " << actual_bits_per_pixel << " bits per pixel provided" << endl;
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{

  struct Params params = getInputParams(argc, argv);
  std::cout << "PIM test: Image Downsampling" << std::endl;

  string input_file = params.inputFile;
  input_file = "./../Dataset/" + input_file;

  // Check if file exists [2]
  struct stat exists_buffer;
  if(stat(input_file.c_str(), &exists_buffer)) {
    std::cout << "Input file \"" << input_file << "\" does not exist!" << endl;
    exit(1);
  }

  std::vector<uint8_t> img = read_file_bytes(input_file);

  if(!createDevice(params.configFile)) {
    return 1;
  }

  if(!check_valid_input_image(img)) {
    return 1;
  }

  vector<uint8_t> pim_averaged = image_downsampling_pim(img);

  if(params.outputFile != nullptr) {
    write_img_to_file(pim_averaged, params.outputFile);
  }

  if(params.shouldVerify) {
    vector<uint8_t> cpu_averaged = image_downsampling_cpu(img);

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

// [2] Vincent, “Fastest way to check if a file exists using standard C++/C++11,14,17/C?,” Stack Overflow, 2024.
// https://stackoverflow.com/a/12774387 (accessed Aug. 07, 2024).
