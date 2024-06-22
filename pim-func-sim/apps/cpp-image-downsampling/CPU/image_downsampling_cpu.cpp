// Image Downsampling implementation on CPU
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

auto current_time_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return nanoseconds;
}

typedef struct Params
{
  char *configFile;
  char *inputFile;
  char *outputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./image_downsampling [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    input image file of BMP type (default=\"input_1.bmp\")"
          "\n    -o    output file for downsampled image (default=no output)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = (char*) "input_1.bmp";
  p.outputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:i:o:")) >= 0)
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
  // Parse BMP file [1]
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
      new_pix.red = (curr_pix1.red>>2) + (curr_pix2.red>>2) + (curr_pix3.red>>2) + (curr_pix4.red>>2);
      new_pix.blue = (curr_pix1.blue>>2) + (curr_pix2.blue>>2) + (curr_pix3.blue>>2) + (curr_pix4.blue>>2);
      new_pix.green = (curr_pix1.green>>2) + (curr_pix2.green>>2) + (curr_pix3.green>>2) + (curr_pix4.green>>2);

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

void write_img(vector<uint8_t>& img, std::string filename) {
    auto outfile = std::fstream(filename, std::ios::out | std::ios::binary);
    outfile.write((char*) img.data(), img.size());
    outfile.close();
}

bool check_image(std::vector<uint8_t>& img) {
  // Verify that input image is of the correct type
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
  std::cout << "CPU test: Image Downsampling" << std::endl;
  
  string input_file = params.inputFile;
  input_file = "../Dataset/" + input_file;
  std::vector<uint8_t> img = read_file_bytes(input_file);

  if(!check_image(img)) {
    return 1;
  }

  auto start_time = current_time_ns();
  vector<uint8_t> cpu_averaged = avg_cpu(img);
  auto end_time = current_time_ns();

  auto dur = ((double) (end_time - start_time))/1000000;

  cout << "Time: " << dur << " ms" << endl;

  if(params.outputFile != nullptr) {
    write_img(cpu_averaged, params.outputFile);
  }
}

// [1] N. Liesch, The bmp file format. [Online]. Available:
// https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm.
