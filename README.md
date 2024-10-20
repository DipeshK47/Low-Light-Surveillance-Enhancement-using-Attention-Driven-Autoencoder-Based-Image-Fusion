# Low-Light Surveillance Enhancement using Attention-Driven Autoencoder-Based Image Fusion

This repository contains the code and methodology for enhancing low-light surveillance images using an attention-driven autoencoder-based image fusion technique. By fusing infrared (IR) and visible spectrum (VIS) images, this model significantly improves the clarity, detail, and detectability of objects in low-light conditions, making it ideal for real-time surveillance applications.

## Table of Contents

	•	Introduction
	•	Problem Statement
	•	Methodology
	•	Architecture
	•	Dataset
	•	Preprocessing
	•	Training
	•	Results
	•	Dependencies
	•	How to Run
	•	Future Work
	•	Contributors
	•	License

## Introduction

Low-light surveillance systems are critical in security infrastructures, especially in urban areas and sensitive locations. However, most existing systems suffer from poor performance in low-light environments, leading to reduced clarity and loss of vital details. This project addresses these challenges using an Attention-Driven Autoencoder-Based Image Fusion technique that integrates the strengths of IR and VIS images, significantly enhancing image quality in low-light settings.

## Problem Statement

Traditional methods like histogram equalization and Retinex-based techniques fail to preserve essential details and often introduce noise in low-light images. Infrared (IR) imaging captures thermal information but lacks visible context, while VIS images struggle in darkness. By fusing these two modalities using the DIDFuse algorithm with an attention mechanism, our approach enhances the clarity, definition, and recognizability of objects under poor lighting conditions. This improvement is essential for security, traffic monitoring, and various other surveillance tasks.

## Methodology

The proposed system integrates an attention mechanism into the DIDFuse algorithm to enhance low-light surveillance images. The methodology includes:

	1.	Feature Extraction: A Convolutional Neural Network (CNN) extracts features from both IR and VIS images.
	2.	Attention Mechanism: Prioritizes discriminative and relevant features from both image modalities.
	3.	Image Fusion: Combines the most informative features from the IR and VIS images, reducing noise and improving clarity.

### Key Techniques:

	•	DIDFuse Algorithm: Fuses infrared and visible spectrum images using CNNs for feature extraction.
	•	Attention Mechanism: Ensures the most discriminative and relevant features are preserved in the final output.
## Dataset

We use the Low Light Visible-Infrared Paired (LLVIP) dataset, which consists of 16,836 pairs of IR and VIS images across 26 different low-light scenes. These scenes capture pedestrians, vehicles, and various objects, providing a diverse set of real-world low-light conditions.

	•	Image Sources: IR and VIS image pairs captured between 18:00 and 22:00, ideal for low-light surveillance applications.
	•	Diversity: Includes various lighting conditions and objects, making it ideal for testing fusion algorithms.
 <div align="center">
<img width="389" alt="Screenshot 2024-10-20 at 4 51 14 PM" src="https://github.com/user-attachments/assets/eb8fae4e-2c8e-44cc-89ff-7b8d48102231">
</div>

## Preprocessing

	•	CLAHE (Contrast Limited Adaptive Histogram Equalization): Used to enhance image contrast and improve visibility in low-light areas.
	•	Normalization: Pixel values are normalized to maintain consistency across lighting conditions.
	•	Image Alignment: Ensures pixel-level alignment of IR and VIS images to avoid blurring and artifacts during fusion.
 
## Architecture

The architecture consists of the following modules:

	1.	Input Module: Low-light IR and VIS image acquisition.
	2.	Preprocessing Module: Image normalization, noise reduction, and pixel-level alignment of IR and VIS images to prevent artifacts during fusion.
	3.	Feature Extraction Module: CNN-based encoder extracts high-level features from both IR and VIS images.
	4.	Attention Mechanism: Enhances critical regions like edges and textures while suppressing less relevant information.
	5.	Fusion Module: Combines the features from both modalities using sum, average, or norm-based fusion strategies.
	6.	Reconstruction Module: A CNN-based decoder reconstructs the final enhanced image.
	7.	Post-Processing Module: Final normalization of the image to prepare it for real-time surveillance applications.
<div align="center">
<img width="476" alt="Screenshot 2024-10-20 at 4 47 25 PM" src="https://github.com/user-attachments/assets/321a1f66-94e3-4fcc-bd32-7592807bdcd1">
</div>

## Training

	1.	Dataset Preparation: The IR and VIS images are preprocessed, normalized, and aligned for training.
	2.	Model Training: The DIDFuse algorithm, enhanced with attention mechanisms, is trained using the prepared dataset.
	3.	Loss Function: The model minimizes the difference between the fused image and the ground truth using custom loss functions.
	4.	Optimizer: The model uses Adam optimizer with a learning rate of 0.001.

## Results

The model achieves notable improvements across several key metrics when tested on the LLVIP dataset:

	•	Entropy (EN): Achieved a score of 5.7566, reflecting significant information gain from the fusion process.
	•	Mutual Information (MI): Scored 2.3634, indicating effective feature integration from IR and VIS images.
	•	Peak Signal-to-Noise Ratio (PSNR): The model achieved a PSNR score of 61.865, indicating high-quality fused images.
	•	Mean Squared Error (MSE): The model recorded an MSE of 0.0428, demonstrating minimal distortion and noise in the fused images.
<div align="center">
  <img width="421" alt="Screenshot 2024-10-20 at 4 45 46 PM" src="https://github.com/user-attachments/assets/d7f3fdbb-1fdf-45aa-9294-a36098b1233f">
</div>
The results show substantial improvement in image quality, noise reduction, and object detection accuracy compared to traditional methods.

## Key Dependencies

	•	Python 3.x
	•	PyTorch
	•	OpenCV
	•	NumPy
	•	Matplotlib
	•	Scikit-learn
	•	TensorBoard (for visualization)

## Future Work

	•	Explore GAN-based approaches to generate even sharper images.
	•	Extend the system to process real-time video streams for live surveillance enhancement.
	•	Investigate unsupervised learning approaches to eliminate the need for ground-truth data during training.

## Contributors

	•	Dipesh Kumar
	•	Aryan Meher
