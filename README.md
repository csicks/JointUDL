# Cryo-EM image alignment: from pair-wise to joint with deep unsupervised difference learning

## Introduction
We provide source code, used data and some additional support for our paper. 
- In folder `Code`, source code is provided for all utilities including network training on nature/synthetic cyro-EM/real-world cryo-EM images, synthetic data generation and so on. Rich comments and instructions are provided.
- In folder `Data`, center images to generate synthetic datasets and sampled MS-COCO images are provided.
- In folder `Additional`，we provide code to generate binary files of well-trained models and show how to use well-trained model/its results in other program such as EMAF.

## Reference
- Please refer to http://www.csbio.sjtu.edu.cn/bioinf/JointUDL/ for used datasets and more information.
- All code is provided for research purpose and without any warranty. If you use code or ideas from this project for your research, please cite our paper: Cryo-EM image alignment: from pair-wise to joint with deep unsupervised difference learning, Yuxuan Chen, Dagan Feng, and Hong-Bin Shen (under review).
- Early version at arXiv is:
	'''
	@article{Chen2022UnsupervisedDL,
	  title={Unsupervised Difference Learning for Noisy Rigid Image Alignment},
	  author={Yu-Xuan Chen and Dagan Feng and Hongbin Shen},
	  journal={ArXiv},
	  year={2022},
	  volume={abs/2205.11829}
	}
	'''