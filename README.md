# hipscreen-ai
Automated Measurement of Migration Percentage in Hip Surveillance Radiographs: Development and Testing of a
Deep-Learning “Artificial Intelligence” Algorithm (Submitted to AACPDM'22 as an abstract)

## Overview
This study aims to develop and test a deep-learning algorithm that automatically measures MP on hip surveillance radiographs. Specifically,two convolutional neural network (CNN) deep-learning models were trained to calculate the MP. The first CNN model based on ResNet 18 architecture was trained to calculate the degrees of rotation needed to level the pelvis, while the second model based on Cascaded Pyramid Network architecture was trained to detect the key landmarks for calculation of MP on the leveled image. The measurement error and the reliability of the deep learning algorithm on the test image set were calculated referenced against expert-labeled “ground truth” MP.