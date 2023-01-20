# AAIT-Assignment2
This repository contains the two tasks for assignment 2 of the AAIT course:
- Image classification with missing labels
- Image classification with noisy labels

## Image classification with missing labels
For this specific task a method based on Pseudo-Labelling is proposed.
1.	Pseudo-labelling - , the proposed network is trained in a supervised fashion with labeled data. After that the same model with the weight learnt in a supervised manner is trained on both labeled and unlabeled data simultaneously. 
2. MixMatch -Stochastic data augmentation is applied to an unlabeled image K times, and each augmented image is fed through the classifier. Then, the average of these K predictions is “sharpened” by adjusting the distribution’s temperature

### Requirements: 
    python==3.10.6
    torch==1.13.0  
    torchvision==0.14.0
    pillow==9.2.0
    matplotlib==3.5.3
