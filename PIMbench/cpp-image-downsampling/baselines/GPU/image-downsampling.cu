// Image Downsampling implementation on GPU
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <sys/stat.h>

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
          "\nUsage:  ./image_downsampling.out [options]"
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
  p.inputFile = (char*) "../../Dataset/input_1.bmp";
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

NewImgWrapper parseInputImageandSetupOutputImage(std::vector<uint8_t> img, bool print_size=false)
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

inline __device__ Pixel* get_pixel(const char* pixels, int scanline_size, int x, int y) {
    return (Pixel*) (pixels + scanline_size*y + 3*x);
}

inline __device__ void set_pixel(const char* pixels, Pixel* new_pixel, int scanline_size, int x, int y) {
    auto* old_pix = (Pixel*) (pixels + scanline_size*y + 3*x);
    *old_pix = *new_pixel;
}

__global__ void imgDSAverage(char* pix_in , char* pix_out , int new_height , int new_width , int old_scanline_size, int new_scanline_size, int new_pixel_data_width) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < new_width) {
        Pixel curr_pix1 = *get_pixel(pix_in, old_scanline_size, 2*x, 2*y);
        Pixel curr_pix2 = *get_pixel(pix_in, old_scanline_size, 2*x + 1, 2*y);
        Pixel curr_pix3 = *get_pixel(pix_in, old_scanline_size, 2*x, 2*y + 1);
        Pixel curr_pix4 = *get_pixel(pix_in, old_scanline_size, 2*x + 1, 2*y + 1);

        Pixel new_pix;
        new_pix.red = (curr_pix1.red>>2) + (curr_pix2.red>>2) + (curr_pix3.red>>2) + (curr_pix4.red>>2);
        new_pix.blue = (curr_pix1.blue>>2) + (curr_pix2.blue>>2) + (curr_pix3.blue>>2) + (curr_pix4.blue>>2);
        new_pix.green = (curr_pix1.green>>2) + (curr_pix2.green>>2) + (curr_pix3.green>>2) + (curr_pix4.green>>2);

        set_pixel(pix_out, &new_pix, new_scanline_size, x, y);
        if(x+1 == new_width) {
            for(int x=new_pixel_data_width; x<new_scanline_size; ++x) {
                pix_out[x] = 0;
            }
        }
    }
}

std::vector<uint8_t> avg_gpu(std::vector<uint8_t> img) {
  
    NewImgWrapper avg_out = parseInputImageandSetupOutputImage(img, true);
    
    char* pixels_out_averaged = (char*) avg_out.new_img.data() + avg_out.new_data_offset;

    char* pixels_in = (char*) img.data() + avg_out.data_offset;


    dim3 dimGrid (( avg_out.new_width + 1023) / 1024 , avg_out.new_height , 1);
    dim3 dimBlock (1024 , 1 , 1);

    char* new_pixels_gpu_average;
    char* old_pixels_gpu;

    int new_pixels_size = avg_out.new_height*avg_out.new_scanline_size;

    cudaError_t errorCode;

    errorCode = cudaMalloc((void**)&new_pixels_gpu_average, new_pixels_size);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMalloc((void**)&old_pixels_gpu, avg_out.old_pixels_size);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    
    errorCode = cudaMemcpy(old_pixels_gpu, pixels_in, avg_out.old_pixels_size, cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    cudaEventRecord(start, 0);

    imgDSAverage<<<dimGrid, dimBlock>>>(old_pixels_gpu, new_pixels_gpu_average, avg_out.new_height, avg_out.new_width, avg_out.scanline_size, avg_out.new_scanline_size, avg_out.new_pixel_data_width);
    
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    printf("Execution time of image downsampling = %f ms\n", timeElapsed);

    errorCode = cudaMemcpy(pixels_out_averaged, new_pixels_gpu_average, new_pixels_size, cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    
    cudaFree(old_pixels_gpu);
    cudaFree(new_pixels_gpu_average);

    return avg_out.new_img;
}

inline Pixel* get_pixel_cpu(const char* pixels, int scanline_size, int x, int y)
{
  return (Pixel*)(pixels + scanline_size * y + 3 * x);
}

inline void set_pixel_cpu(const char* pixels, Pixel* new_pixel, int scanline_size, int x, int y)
{
  auto* old_pix = (Pixel*)(pixels + scanline_size * y + 3 * x);
  *old_pix = *new_pixel;
}

std::vector<uint8_t> avg_cpu(std::vector<uint8_t> img)
{
  //    Averaging Kernel
  NewImgWrapper avg_out = parseInputImageandSetupOutputImage(img);
  char* pixels_out_averaged = (char*)avg_out.new_img.data() + avg_out.new_data_offset;
  char* pixels_in = (char*)img.data() + avg_out.data_offset;

  for (int y = 0; y < avg_out.new_height; ++y) {
    for (int x = 0; x < avg_out.new_width; ++x) {  // 4 per get pixel
      Pixel curr_pix1 = *get_pixel_cpu(pixels_in, avg_out.scanline_size, 2 * x, 2 * y);  // 4 + 2
      Pixel curr_pix2 = *get_pixel_cpu(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y);  // 4 + 3
      Pixel curr_pix3 = *get_pixel_cpu(pixels_in, avg_out.scanline_size, 2 * x, 2 * y + 1);  // 4 + 3
      Pixel curr_pix4 = *get_pixel_cpu(pixels_in, avg_out.scanline_size, 2 * x + 1, 2 * y + 1);  // 4 + 4

      Pixel new_pix;
      new_pix.red = (curr_pix1.red>>2) + (curr_pix2.red>>2) + (curr_pix3.red>>2) + (curr_pix4.red>>2);
      new_pix.blue = (curr_pix1.blue>>2) + (curr_pix2.blue>>2) + (curr_pix3.blue>>2) + (curr_pix4.blue>>2);
      new_pix.green = (curr_pix1.green>>2) + (curr_pix2.green>>2) + (curr_pix3.green>>2) + (curr_pix4.green>>2);

      set_pixel_cpu(pixels_out_averaged, &new_pix, avg_out.new_scanline_size, x, y);
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

  cudaGetLastError();

  struct Params params = getInputParams(argc, argv);
  std::cout << "GPU test: Image Downsampling" << std::endl;
  
  string input_file = params.inputFile;

  // Check if file exists [2]
  struct stat exists_buffer;
  if(stat(input_file.c_str(), &exists_buffer)) {
    std::cout << "Input file \"" << input_file << "\" does not exist!" << endl;
    exit(1);
  }

  std::cout << "Input file: '" << input_file << "'" << std::endl;
  std::vector<uint8_t> img = read_file_bytes(input_file);

  if(!check_image(img)) {
    return 1;
  }

  vector<uint8_t> gpu_averaged = avg_gpu(img);

  if(params.outputFile != nullptr) {
    write_img(gpu_averaged, params.outputFile);
  }

  if(params.shouldVerify) {
    vector<uint8_t> cpu_averaged = avg_cpu(img);

    if (cpu_averaged.size() !=gpu_averaged.size()) {
      cout << "Average kernel fail, sizes do not match" << endl;
      return 1;
    }
    for (size_t i = 0; i < cpu_averaged.size(); ++i) {
      if (cpu_averaged[i] != gpu_averaged[i]) {
        cout << "Average kernel mismatch at byte " << i << endl;
        return 1;
      }
    }

    cout << "GPU Result matches CPU result" << endl;
  }
}

// [1] N. Liesch, The bmp file format. [Online]. Available:
// https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm.


// [2] Vincent, “Fastest way to check if a file exists using standard C++/C++11,14,17/C?,” Stack Overflow, 2024.
// https://stackoverflow.com/a/12774387 (accessed Aug. 07, 2024).