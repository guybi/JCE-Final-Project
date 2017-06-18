# Deep Learning for Medical Images

This project was carried out as part of the studies at JCE.

Automatic organ classification is an important and challenging problem for medical image analysis.
In recent years, techniques have been developed for various classification problems in many fields, especially in medicine. Most techniques are based on artificial neural networks, and in particular artificial neural networks with convolution layers, which are better in image analysis.
 
In this project, we present a solution for classification of organs from CT scans using artificial neural networks (ConvNets).
The center of the project focuses on investigating the problem of classification by neural networks.
We have a CT dataset, which our goal if to build a system that claims CT scans so that the system can analyze and obtain useful information from scans for classification purposes.
We will detail the neural networks in general, detailing the stages of preparation of the information and the stages in building the neuron network, detailing the network we created, the training process and the results we received, as well as details of the difficulties that arose during the development stages.


## Table of Contents

- [Preparations](#preparations)
- [Usage](usage)

## Preparations
1. Install Python 3.5
2. Install these depenedencies:
    - tensorflow (or tensorflow-gpu for gpu calculation support)
    - numpy
    - simpleitk
    - pillow
3. If you use tensorflow-gpu, you will need to install also:
    - NVIDIA CUDA
    - NVIDIA cuDNN

## Usage

1. Clone this repository.
2. Ensure you installed all the required dependecies (as mentioned in preparations paragraph).
3. Ensure you downloaded "anatomy3-benchmark database". Note that if you are using another dataset, you probably need to do code optimizations for this dataset.
4. For start training and data preparation, run "run_train.py" script.
5. For testing your model (weights), run "test_model.py" script.
