# Cryo-EM image alignment: from pair-wise to joint with deep unsupervised difference learning
[![DOI](https://img.shields.io/badge/doi-10.1021/acs.jcim.1c00745-blue.svg)](https://doi.org/10.1021/acs.jcim.1c00745)

## Introduction
We provide source code, used data and some additional support for our paper. 
- In folder `Code`, source code is provided for all utilities including network training on nature/synthetic cyro-EM/real-world cryo-EM images, synthetic data generation and so on. Rich comments and instructions are provided.
- In folder `Data`, center images to generate synthetic datasets and real-world cryo-EM datasets are provided. For real-world cryo-EM datasets, please refer to our website.
- In folder `Additional`，we provide code to generate binary files of well-trained models and show how to use well-trained model/its results in other program such as EMAF.
- In folder `Natural Images`，we provide code and data for UDL on natural images from Microsoft COCO dataset. Though this part is not our goal and not introduced in the paper, UDL could get promising results for rigid noisy image alignment while STN-based methods generally do not converge.
- Trained models as initialization for training (only UDL model) are provided on our website. Such initialization generally leads to better convergence but if not, multiple training/fine tuning might be required. You may also train from the very beginning on one synthetic dataset (we use GroEL) and generate your own initialization.

## License
- Our code/program is licensed under the GNU General Public License v3.0; you may not use this file except in compliance with the License. See file LICENSE in folder `Code` and `Natural Images` for details.
- All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode 


## Reference
- Please refer to http://www.csbio.sjtu.edu.cn/bioinf/JointUDL/ for used datasets and more information.
- All code is provided for research purpose and without any warranty. If you use code or ideas from this project for your research, please cite our paper: 
    ```bibtex
    @article{CHEN2023107940,
    title = {Cryo-EM image alignment: From pair-wise to joint with deep unsupervised difference learning},
    journal = {Journal of Structural Biology},
    volume = {215},
    number = {1},
    pages = {107940},
    year = {2023},
    issn = {1047-8477},
    doi = {https://doi.org/10.1016/j.jsb.2023.107940},
    url = {https://www.sciencedirect.com/science/article/pii/S1047847723000035},
    author = {Yu-Xuan Chen and Dagan Feng and Hong-Bin Shen},
    keywords = {Cryo-EM image alignment, Unsupervised learning, Difference learning, Joint alignment}
    }
    ```
- Early version at arXiv is at https://arxiv.org/abs/2205.11829 with bibtex:
	```bibtex
	@article{Chen2022UnsupervisedDL,
	  title={Unsupervised Difference Learning for Noisy Rigid Image Alignment},
	  author={Yu-Xuan Chen and Dagan Feng and Hongbin Shen},
	  journal={ArXiv},
	  year={2022},
	  volume={abs/2205.11829}
	}
	```