# Sample images for MS-COCO dataset

## Introduction
This folder contain sampled images from MS-COCO dataset, which is used to show file strctures of dataset folder used to generate synthetic MS-COCO dataset for rigid image alignment. Due to space limit of Supplementary Files, we cannot upload the entire MS-COCO dataset but readers may organize MS-COCO dataset folder in the following file structure (which is also the original file structe of MS-COCO dataset after unzipping) and generate synthetic MS-COCO dataset for rigid image alignment.

By directly pass path of folder in this file strcture to file `code/synthetic/data_coco.py`, synthetic MS-COCO dataset can be generated. Note that pathches are randomly extracted from images (one patch from one image) and patches with too little details will be disgarded. In our experiments, we get about 115k image patch pairs from `train` folder and 5k image patch pairs from `val` folder.
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

## References
- Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.