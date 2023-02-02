# Cryo-EM image alignment: from pair-wise to joint with deep unsupervised difference learning

## Generating datasets
- File `data_coco.py` is used to generate synthetic MS-COCO dataset for rigid image alignment. The input folder should be in file structure shown below. See code and comments for details.
  ```
  MS-COCO samples
  │ 
  │
  └───train2017
  │   │   image_t1.png
  │   │   image_t2.png
  │   │   ...
  │   
  └───val2017
      │   image_v1.png
      │   image_v2.png
      │   ...
  ```
- File `coco.py` is used for MS-COCO dataset in PyTorch.

## Training and Testing
- File `main_coco_clean.py`, `main_coco_gaussian.py` and `main_coco_sp.py` are used for training/testing on synthetic MS-COCO dataset for UDL-based unsupervised pair-wise alignment method. These three files correspond to clean images, Gaussian noisy images and salt-and-pepper noisy images with only parameter `label` replaced in main function. See code and comments for details.
- File `main_scoco_clean.py`, `main_scoco_gaussian.py` and `main_scoco_sp.py` are used for training/testing on synthetic MS-COCO dataset for supervised pair-wise alignment method. These three files correspond to at clean images, Gaussian noisy images and salt-and-pepper noisy images with only parameter `label` replaced in main function. See code and comments for details.
- File `main_stn.py` is used for training/testing on synthetic MS-COCO dataset for STN-based unsupervised pair-wise alignment method. This file aims at clean images. See code and comments for details.

## References
- Mask module: Zhang, Jirong, et al. "Content-aware unsupervised deep homography estimation." European Conference on Computer Vision. Springer, Cham, 2020.
- Feature extraction module: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Feature matching module: Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- Feature matching module: Zeng, Xiangrui, and Min Xu. "Gum-Net: Unsupervised geometric matching for fast and accurate 3D subtomogram image alignment and averaging." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
