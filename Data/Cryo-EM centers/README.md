# Picked particles for generating synthetic cryo-EM dataset

## Introduction

This folder contain class averages for generating synthetic cryo-EM dataset, with file structure shown below. These seven class averages are correspondingly extracted from cluster centers generated from real-world GroEL dataset, EMPIAR-10034 dataset, EMPIAR-10398 dataset, EMPIAR-10029 dataset, EMPIAR-10065 dataset, EMPIAR-10874 dataset and EMPIAR-10397 dataset by cryo-EM clustering method CL2D. These seven dataset are already CTF corrected for both phase and amplitude by software Scipion before feed into clustering method. By directly pass path to this folder to file `code/synthetic/data_em.py`, synthetic cryo-EM single particle dataset can be generated.
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

## References

- GroEL: Ludtke, Steven J., et al. "Seeing GroEL at 6 Å resolution by single particle electron cryomicroscopy." Structure 12.7 (2004): 1129-1136.
- EMPIAR: Iudin, Andrii, et al. "EMPIAR: a public archive for raw electron microscopy image data." Nature methods 13.5 (2016): 387-388. 
- CL2D: Sorzano, Carlos Oscar S., et al. "A clustering approach to multireference alignment of single-particle projections in electron microscopy." Journal of structural biology 171.2 (2010): 197-206.
- Scipion: De la Rosa-Trevín, J. M., et al. "Scipion: A software framework toward integration, reproducibility and validation in 3D electron microscopy." Journal of structural biology 195.1 (2016): 93-99.