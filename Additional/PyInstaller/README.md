# Binary File Generation using PyInstaller

## Introduction
This folder contains code used to binary file generation for UDL/JUDL using PyInstaller package. We have tested PyInstaller under CUDA version 10.2 in Ubuntu 16.04 with packages listed in `requirements_CUDA10.2.txt` in this folder. Actually, the required package are PyTorch, OpenCV, mrcfile and PyInstaller. You may simply install these packages using Conda step by step.


## Problems
Some encountered problems are listed here:
- When you try to generate generation for UDL/JUDL, please generate the binary file into a single folder (`pyinstaller -D xx.py`) instead of a single file (`pyinstaller -F xx.py`). 
- When organizing the environment, please install all packages using Conda and do not use Conda and Pypi in the same time.
- When encountering errors like missing file `xxx.so`, you may need to manually move these files to the corresponding position in the generated folder.You may find `xxx.so` in folders of Conda environment.