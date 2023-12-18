# Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation

ECE-GY 9143 - High Performance Machine Learning 

Final Project

by Team 3  - Raghav Rawat (rr3418) & Utkarsh Prakash Srivastava (ups2006) 

## Description

The primary goal of this project is to advance in medical imaging through the application of advanced deep learning techniques, specifically focusing on 3D segmentation of brain tumors using U-Net architecture [1]. The project aim to implement and optimize HPC concepts, including data parallelism, to harness the computational power of parallel processing units. This will ensure swift training and experimentation with the large-scale dataset, allowing for rapid model iteration and evaluation.


One of the significant challenges in healthcare is accurate and timely diagnosis. By developing advanced 3D segmentation techniques for brain tumor detection, this project contributes to precise medical diagnosis. Early detection and accurate segmentation enable healthcare professionals to provide personalized and effective treatment plans, improving patient outcomes and quality of life. We focus on achieving this utilizing high-performance computing (HPC) concepts such as transfer learning. This ensures that the computational resources are used efficiently, making medical image analysis more accessible and cost-effective. Scalability allows the application of these techniques in various healthcare settings, including resource-limited environments, expanding the reach of quality healthcare services. Training deep learning models with limited data is a common challenge. By applying transfer learning techniques, the knowledge is transferred to train on smaller datasets effectively. This approach democratizes access to advanced medical image analysis tools, allowing medical professionals in diverse regions to leverage cutting-edge technologies for accurate diagnosis and treatment planning.

### Dataset Description

The BraTS 2020 dataset [2-4], comprising high-quality 3D MRI images with meticulously annotated sub-regions, is the primary data source. It comprises high-quality 3D MRI images with meticulously annotated sub-regions. Developed as part of the Multimodal Brain Tumor Segmentation Challenge, the Brats 2020 dataset incorporates high-quality magnetic resonance imaging (MRI) scans. It included files in .nii format (Neuroimaging Informatics Technology Initiative files - NIfTI), which stores volumetric data, such as three-dimensional images obtained from imaging modalities like magnetic resonance imaging (MRI) or computed tomography (CT). For each patient, we are provided with different modalities, such as T1-weighted, T1-weighted contrast-enhanced, T2-weighted, and FLAIR (Fluid Attenuated Inversion Recovery), offering a multi-faceted view of brain structures and abnormalities.  

![images/patient1.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/dcbf2e0018733b145c039acc7ba110c3958f39bd/images/patient1.png)

Figure 1 representing T1, T2, T1ce, Flair and Mask Slices for patient001.

### Model Details

We employ the U-Net architecture, specifically designed for biomedical image segmentation, as the foundational model. Figure 2 represents the architecture design of a 3D Unet model. The 3D UNet architecture represents a significant advancement in the field of medical image segmentation. Unlike its 2D counterparts, this architecture operates on entire three-dimensional volumes, allowing it to capture intricate spatial relationships and contextual information crucial for accurate segmentation in medical imaging tasks. The 3D UNet architecture is structured into distinct - analysis and synthesis paths. In the analysis path, each layer consists of two 3×3×3 convolutions, followed by a ReLU activation, and a 2×2×2 max pooling with strides of two in each dimension. Conversely, the synthesis path employs a 2×2×2 up-convolution with strides of two, followed by two 3×3×3 convolutions and ReLU activations \cite{desc}. One of its key strengths lies in the incorporation of skip connections, facilitating the direct flow of high-resolution features from the encoder to the corresponding layers in the decoder. These connections enable the network to precisely localize and segment objects within the input volume. The final layer produces a segmentation mask, assigning labels to each voxel based on the object or region it represents. This innovative approach has proven highly effective in applications such as brain tumor segmentation and organ delineation, making it a cornerstone in cutting-edge 3D medical image analysis.

![images/3dunet.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/0aeea6423cdd13ad877d603408292103770600b0/images/3dunet.png)

Figure 2: Architecture design of 3D U-Net architecture.


## References 
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-NET: Convolutional Networks for Biomedical Image Segmentation,” in Lecture Notes in Computer Science, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)




