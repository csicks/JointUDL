# EMAF: Fast Cryo-EM Image Alignment Algorithm using Power Spectrum Features

## Introduction
- EMAF is an alignment algorithm program which aims at cryo-EM two-dimensional particle images. The program is implemented in C++ under C++14 standard. Some other alignment methods like Fourier-Mellin transform and XMIPP alignment are also re-implemented and included in this program. Both source code and complied program are available.
- This program has been tested on Ubuntu 14.04, Ubuntu 16.04 and Ubuntu 18.04. Other Linux system should also be able to run this program. For Windows, you may be able to use this program after installation of FFTW.

## Dependencies
- Since most utilities are implemented in our own source code, the only dependent package is FFTW. You may simply install FFTW3 using `sudo apt-get install libfftw3-dev` in ubuntu command line or compiling from source code following instructions on the [official website](http://fftw.org/).

## Usage
- You may compile from source code using the following command in Linux system. CMake version 3.10 and above have been tested. You may change the [CMake File](./CMakeLists.txt) if other CMake version are used.

  ```bash
  cd /path/to/README
  mkdir build
  cd build
  cmake ..
  make
  ```

- By now, only MRC format and certain XMD format are supported as input file. XMD format is used for XMIPP relied programs and only tested under certain cases. Therefore, data is strongly recommended to be converted to MRC format when it is feed into this program.

## Suggestions
- In this version, `main.cpp` is modified to use output results of UDL/JUDL. The original version of `main.cpp` is renamed as `main_backup.cpp` in this folder.
- In this version, the complied binary file and some information in `README` is removed. Please refer to http://www.csbio.sjtu.edu.cn/bioinf/EMAF/ for used datasets and more information.

## Others
- All code is provided for research purpose and without any warranty. If you use  code or ideas from this project for your research, please cite our paper:
  ```
  @Article{Chen2021a,
  author    = {Chen, Yu-Xuan and Xie, Rui and Yang, Yang and He, Lin and Feng, Dagan and Shen, Hong-Bin},
  journal   = {J. Chem. Inf. Model.},
  title     = {Fast Cryo-EM Image Alignment Algorithm Using Power Spectrum Features},
  year      = {2021},
  issn      = {1549-9596},
  month     = sep,
  number    = {9},
  pages     = {4795--4806},
  volume    = {61},
  comment   = {doi: 10.1021/acs.jcim.1c00745},
  doi       = {10.1021/acs.jcim.1c00745},
  publisher = {American Chemical Society},
  url       = {https://doi.org/10.1021/acs.jcim.1c00745},
  }
  ```