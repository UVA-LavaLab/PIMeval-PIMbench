# Brightness

Brightness is a basic image processing algorithm that increments the color channel components for all image pixels by a specified degree. The PIM, CPU, and GPU implementations all use the same process of accomplishing this adjustment, by adding the brightness coefficient to each pixel of image data, peforming a check to see if the change goes out of bounds for valid pixel values. If out of bounds situations occur, the pixel is rounded to the nearest max/min value, such as 276 rounding down to 255 or -20 rounding up to 0. The operation is defined as:

$ \forall p \in Image , \space p[r,g,b] \space \texttt{+}\texttt{=} \space BC \space | \space 0 \space \texttt{<}\texttt{=} \space p_i[r,g,b] \space \texttt{<}\texttt{=} \space 255$

where:
- $p$ is a pixel in the $Image$ input
- $r,g,b$ are the red, green, and blue color channels of a pixel
- $BC$ is the brightness coefficient, set by the user

For a detailed description of the Brightness algorithm, refer to the OpenCV brightness [documentation](https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html).

## Input Files

As mentioned above, only 24-bit .bmp files can be used to test the functionality of this benchmark. Sample inputs can be found in the `/cpp-histogram/histogram_datafiles/` directory, which were gathered from the Phoenix [GitHub](https://github.com/fasiddique/DRAMAP-Phoenix/tree/main) ([direct download link](http://csl.stanford.edu/~christos/data/histogram.tar.gz)), with the execepton of `sample1.bmp`, which came from [FileSamplesHub](https://filesampleshub.com/format/image/bmp). Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/u/3/folders/1sKFcEftxzln6rtjftChb5Yog_9S5CDRd).

## Compilation Instructions

To compile, simply run:

```bash
$ make 
```

## Execution Instructions

After compiling, run the executable using the following command:

```bash
$ ./brightness.out
```

`/cpp-histogram/histogram_datafiles/sample1.bmp` is used as the default input file; however, you can also specify a valid .bmp file using the `-i` option:

```bash
$ ./brightness.out -i <input_file>
```

If you want to change the brightness coefficient (i.e. the amount the color channel values change by), use the `-b` option:

```bash
$ ./brightness.out -b <value>
```

If answer verification with a baseline CPU run is desired, use the `-v` flag:

```bash
$ ./brightness.out -v t
```

The cases described above and more can be used simultaneously. To see more details regarding the usage, use the `-h` option:

```bash
$ ./brightness.out -h
```