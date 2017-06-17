# Deep Learning for Medical Images

This project was carried out as part of the studies at JCE.

## Table of Contents

- [Preparations](#preparations)
- [Usage](usage)

## Preparations
1. Install Python 3.5
2. Install these depenedencies:
  - tensorflow (or tensorflow-gpu for gpu calculation support)
  - numpy
  - simpleitk
  - 

## Usage

1. Clone this repository.
2. Ensure you installed all the required dependecies (as mentioned in preparations paragraph).
3. Ensure you downloaded "anatomy3-benchmark database". Note that if you are using another dataset, you probably need to do code optimizations for this dataset.
4. For start training and data preparation, run "run_train.py" script.
5. For testing your model (weights), run "test_model.py" script.
