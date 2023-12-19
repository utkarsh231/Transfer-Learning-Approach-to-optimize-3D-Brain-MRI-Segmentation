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

_Figure 1 representing T1, T2, T1ce, Flair and Mask Slices for patient001._

### Model Details

We employ the U-Net architecture, specifically designed for biomedical image segmentation, as the foundational model. Figure 2 represents the architecture design of a 3D Unet model. The 3D UNet architecture represents a significant advancement in the field of medical image segmentation. Unlike its 2D counterparts, this architecture operates on entire three-dimensional volumes, allowing it to capture intricate spatial relationships and contextual information crucial for accurate segmentation in medical imaging tasks. The 3D UNet architecture is structured into distinct - analysis and synthesis paths. In the analysis path, each layer consists of two 3×3×3 convolutions, followed by a ReLU activation, and a 2×2×2 max pooling with strides of two in each dimension. Conversely, the synthesis path employs a 2×2×2 up-convolution with strides of two, followed by two 3×3×3 convolutions and ReLU activations \cite{desc}. One of its key strengths lies in the incorporation of skip connections, facilitating the direct flow of high-resolution features from the encoder to the corresponding layers in the decoder. These connections enable the network to precisely localize and segment objects within the input volume. The final layer produces a segmentation mask, assigning labels to each voxel based on the object or region it represents. This innovative approach has proven highly effective in applications such as brain tumor segmentation and organ delineation, making it a cornerstone in cutting-edge 3D medical image analysis.

![images/3dunet.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/0aeea6423cdd13ad877d603408292103770600b0/images/3dunet.png)

_Figure 2: Architecture design of 3D U-Net architecture._

## Project milestones 

This project is structured around four key milestones: Data Preparation, Model Training, Weights Download and Transfer, and Training on a New Dataset.

### Data Preparation

The Data Preparation involves several steps to prepare the dataset ready for training. The data preparation began with reading the BraTS 2020 dataset which comprised of high-quality 3D MRI images with meticulously annotated sub-regions. It included files in .nii format (Neuroimaging Informatics Technology Initiative files - NIfTI), which stores volumetric data, such as three-dimensional images obtained from imaging modalities like magnetic resonance imaging (MRI) or computed tomography (CT). To use the dataset properly for 3D-segmentation, different modalities of a patient needed to be combined together before feeding it to the model. For the initial state of model training, modalities FLAIR and T1CE were zipped together using data generator. The dataset was then divided into three categories - train, validation and test as shown in figure 3.

![images/split.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/646ee6198251984c69152fd4e035adea6721a3ba/images/split.png)

_Figure 3: Bar Graph representing distribution of training, validation and testing data subsets._

### Domain Shift Measurement Function
![images/PCA%20analysis.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/main/images/PCA%20analysis.png)


![images/TSNE%20analysis.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/main/images/TSNE%20analysis.png)

Inputs:

**X and Y**: Numpy arrays representing two sets of samples. They have shapes (n_samples, n_features).
kernel: The kernel function to be used, with options for 'rbf' (Radial basis function), 'linear', etc.
gamma: Parameter for the 'rbf' kernel. It is ignored for other kernels.
batch_size: Batch size for batch-wise computation of kernel matrices.
Outputs:

**mmd**: Maximum Mean Discrepancy between the two sets of samples.
Details:

The function iterates over the samples in batches for both sets (X and Y).
For each batch, it computes the kernel matrices (K_XX, K_YY, and K_XY) based on the chosen kernel function.
If the kernel is 'rbf', the Radial Basis Function (RBF) kernel is computed using the rbf_kernel function.
If the kernel is 'linear', the linear kernel is computed using dot products.
The MMD is then updated based on these kernel matrices.
The final MMD is the sum of contributions from all batches.

### Model Training
The dataset then, was fed into 3D Unet architecture for training. The model was trained on 35 epochs. Adam optimizer was used to train the model with  categorical crossentropy as loss function. The performance was measured on following evaluation metrics - accuracy, loss, precision, sensitivity (a measure of the ability of a segmentation algorithm to correctly identify positive instances), specificity (a measure of the ability of a segmentation algorithm to correctly identify negative instances), dice coefficient (a measure of the spatial overlap between two binary images), and mean iou (it measures the similarity between the predicted segmentation and the ground truth segmentation of an image).  Figure 4 represents tabular scores obtained after training the model. The weights of the model is then saved in an .h5 file for using the further in the project. Figure 5 represents training and validation loss and accuracy plots during training.

![images/results%20after%20training.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/c03149ae6fa1078eb6a53eebb5394b173cbc6134/images/results%20after%20training.png)

_Figure 4: Training and Validation scores after training the model._

![images/traingraph.png](https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/c03149ae6fa1078eb6a53eebb5394b173cbc6134/images/traingraph.png)

_Figure 5: Plots representing Training vs validation accuracy and loss curve._

### Weights download and transfer learning

We propose an intriguing approach of transfer learning among the MRI Modalities to help in the domain generalization of the final model. Different MRI modalities refer to various imaging sequences or pulse sequences that emphasize different tissue contrasts or highlight specific characteristics. We first train the model on the 'T1-FLAIR' and 'T1CE' zipped together, and later freeze learnt weights. In the second phase of our work we hypothesize the transfer of these learnt weights to be fine tune them further with other two modalities - 'T1 weighted' and 'T2 weighted'. This helps us reduce the training data in one go and hence improving the training metrics. 

### Training on new dataset
For tuning the final layers of the model on 'T1 weighted' and 'T2 weighted' there are some considerations to be made to ensure that the utility of transfer learning is overall beneficial and justified. Some of these include - 

**Domain Shift**:
If there is a significant domain shift between the training modalities and the target modalities, the effectiveness of transfer learning may be reduced. Domain adaptation techniques might be necessary to mitigate this challenge. We can employ methods like t-SNE, Domain Confusion Loss and covariance shift to measure the shifts. 

**Data Availability**:
Sufficient labeled data for the target modalities is crucial for effective transfer learning. If there is a lack of labeled data for the target modalities, it might be challenging to adapt the model successfully.

**Shared Information**:
Transfer learning assumes that the model has learned useful representations that are applicable across different modalities. If the information learned from the initial two modalities is not transferable or relevant to the other two modalities, the effectiveness of transfer learning may be limited. This bounds us to use the same set of patients having the data for all four modalities so that they can be consistent in the two datasets involved.

One example in the medical imaging domain where transfer learning has been applied across different modalities involves the use of pre-trained models on natural images to improve performance on medical image analysis tasks. The characteristics of natural images and medical images can differ significantly. However, the low-level features learned during pre-training can be useful in extracting relevant patterns in medical images. In such tasks domain adaptation might still be required to ensure that the transfer learning process does not reduce the performance. However, in our use case, since we target to use modalities of the same patient groups, the domain does not differ by large and hence there is a definitive improvement in performance by transfer of learnt features. 

## Description of the repository

The repository contains:- 

1. **get_data_ready.ipynb** : This ipynb file contains the code responsible for initial part of the project, ie, reading the dataset, visualizing and defining and training on 3D UNet architecture. The file also returns "model_x1_1.h5" file, which contains the saved model parameters. This file will be transferred and used for the next part of the project.
2. **domain-shift-analysis.ipynb**: The function `maximum_mean_discrepancy` calculates the domain shift between two modalities used for UNet Training (T1CE and Flair) vs the two modalities that will be used for fine-tuning (T1 and T2). 
3. **TransferLearning.ipynb** : This file trains the dataset using the generated h5 file in previous part. The certain amount of initial layers are frozen. The model trains on different modalities than those used before on get_data_ready.ipynb.
4. **Profiling Folder**: The folder is further divided into 2 subfolders which contains profiling json files obtained during training first(get_data_ready.ipynb) and latter part(TransferLearning.ipynb). The folders also contains screenshot from different instances.
## Example commands to execute the code 
Download the .ipynb file and run all the cells.

## Results


<div style="text-align:center;">
  <img src="https://github.com/utkarsh231/Transfer-Learning-Approach-to-optimize-3D-Brain-MRI-Segmentation/blob/main/images/results.png" width="520">
</div>

## References 
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-NET: Convolutional Networks for Biomedical Image Segmentation,” in Lecture Notes in Computer Science, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)




