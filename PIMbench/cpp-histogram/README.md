# Histogram

Histogram is a basic image processing algorithm that analyzes the number of instances that fall into a predetermined number of bins. For our use case, we analyze 24-bit .bmp files and create 256 bins for 3 separate channels. The number of bins is significant as they represent each key value from 0 to 255, while the channels corresponds to either red, green, or blue (RGB) color characteristics within the image. The .bmp file format is chosen as data is easy to extract and perform arithmetic on. Additionally, previous works, such as Phoenix [[1]](#1), use the same format and implementation. This PIM implementation uses sequential memory accessing, instead of the random accessing that is utilized in the CPU and GPU benchmarks which use threading and CUB respectively.

## Input Files

As mentioned above, only 24-bit .bmp files can be used to test the functionality of this benchmark. Sample inputs can be found in the `~/cpp-histogram/histogram_datafiles/` directory, which were gathered from the Phoenix [GitHub](https://github.com/fasiddique/DRAMAP-Phoenix/tree/main) ([direct download link](http://csl.stanford.edu/~christos/data/histogram.tar.gz)), with the execepton of `sample1.bmp`, which came from [FileSamplesHub](https://filesampleshub.com/format/image/bmp). Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/u/3/folders/1sKFcEftxzln6rtjftChb5Yog_9S5CDRd).

## Compilation Instructions

To compile, simply run:

```bash
$ make 
```

## Execution Instructions

After compiling, run the executable using the following command:

```bash
$ ./hist.out
```

`/histogram_datafiles/sample1.bmp` is used as the default input file; however, you can also specify a valid .bmp file using the `-i` flag:

```bash
$ ./hist.out -i <input_file>
```

If answer verification with a baseline CPU run is desired, use the `-v` flag:

```bash
$ ./hist.out -v t
```

The cases described above and more can be used simultaneously. To see more details regarding the usage, use the `-h` option:

```bash
$ ./hist.out -h
```

## References

<a id = "1">[1]</a>
Colby Ranger, Ramanan Raghuraman, Arun Penmetsa, Gary Bradski,
and Christos Kozyrakis. Evaluating mapreduce for multi-core and
multiprocessor systems. In 2007 IEEE 13th International Symposium
on High Performance Computer Architecture, pages 13–24. Ieee, 2007.
