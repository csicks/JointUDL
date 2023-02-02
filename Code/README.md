# Cryo-EM image alignment: from pair-wise to joint with deep unsupervised difference learning

## Introduction
This program aims to deal with unsupervised rigid image alignment. The unsupervised learning strategy is based on our proposed Unsupervised Difference Learning (UDL) and the network architecture is based on ResNet. Joint alignment is also introduced on the basis of UDL. This program adopts previous works like feature matching, siamese configuration, mask module and etc. The whole program is implemented in PyTorch. Rich comments are included in each code file.

## Environments
The whole program is implemented in Python3 using PyTorch package. The Python version used in experiments is 3.7.10. CUDA version 10.2 and 11.3 has been tested, with corresponding installer packages listed in file `requirements_CUDA10.2.txt` and `requirements_CUDA11.3.txt` which may be used to create Python environment in Conda. Note that the code should be executed under CUDA environment using GPU. If CPU is preferred, some code may need to be revised in source code.

## Generating datasets
- Code files related to generating datasets are located in folder `synthetic`. Sampled MS-COCO datasets to show file structure and cryo-EM picked particles are included in folder `Data`.
- File `data_em.py` and file `data_em_ssnr.py` are used to generate synthetic cryo-EM dataset for rigid image alignment. The input folder should be in file structure shown below.
  ```
  Cryo-EM centers
  │ 
  │
  └───1
  │   │   ori.png (from GroEL)
  │   
  └───2
  │   │   ori.png (from EMPIAR 10034)
  │   
  └───3
  |   │   ori.png (from EMPIAR 10398)
  │   
  └───4
  |   │   ori.png (from EMPIAR 10029)
  │   
  └───5
  │   │   ori.png (from EMPIAR 10065)
  │   
  └───6
  │   │   ori.png (from EMPIAR 10874)
  │   
  └───7
      │   ori.png (from EMPIAR 10397)
  ```


## Training and Testing
- File `main_synthetic_udl.py` is used for training/testing on synthetic cryo-EM dataset for UDL-based unsupervised pair-wise alignment method. See code and comments for details.
- File `main_real_udl.py` is used for training/testing on real-world cryo-EM dataset GroEL for UDL-based unsupervised pair-wise alignment method. See code and comments for details.
- File `main_synthetic_judl.py` is used for training/testing on synthetic cryo-EM dataset for UDL-based unsupervised joint alignment method. See code and comments for details.
- File `main_real_judl.py` is used for training/testing on real-world cryo-EM dataset GroEL for UDL-based unsupervised joint alignment method. See code and comments for details.
- File `main_mrc32.py` is used for generating alignment results on data using well-trained UDL/JUDL model. MRCS file (recommended) and certain XMD file are supported as inputs. Batch size 32 is used. See code and comments for details (alignment parameters and aligned image stack in MRCS format).
- File `evaluate.py` is used for evaluate results on data when compared to reference image (the first image is used as reference image in implementation) using well-trained UDL/JUDL model. See code and comments for details.
- File `evaluate_pair.py` is used for evaluate results on data when compared pair-wise (compared to the next image in dataset in implementation) using well-trained UDL/JUDL model. See code and comments for details.

## References
- Mask module: Zhang, Jirong, et al. "Content-aware unsupervised deep homography estimation." European Conference on Computer Vision. Springer, Cham, 2020.
- Feature extraction module: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Feature matching module: Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- Feature matching module: Zeng, Xiangrui, and Min Xu. "Gum-Net: Unsupervised geometric matching for fast and accurate 3D subtomogram image alignment and averaging." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
