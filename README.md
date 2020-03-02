# M2P3
  This repository contains the code for the paper "M2P3: Multimodal Multi-Pedestrian Path Prediction by Self-Driving Cars With Egocentric Vision"
  https://www.dfki.de/~klusch/i2s/Paper_1137_-_M2P3.pdf	

# Dependencies
  
  The code was tested on Ubuntu 16.04,Python 3 and a GTX 1080ti gpu . The following dependencies are needed:
  ```
  numpy
  scipy
  Pillow
  cython
  matplotlib
  scikit-image
  tensorflow>=1.3.0
  keras>=2.0.8
  opencv-python
  h5py
  imgaug
  scikit-learn
  ```
  The dependencies can be installed by using "pip install"


# Instructions
  
  Train/test a model on the JAAD dataset. The model is currently using just the past bounding boxes of the pedestrians to make a prediction. The model observes 0.5 seconds in the past and predicts 1 second into the future.

  1. To train a model run:
  
  ```
   python m2p3.py --train
  ```
   This will train a model with the default hyperparameters and will save the model in the models/ folder.
  
  2. To test and visualize a model run:

  ```
   python m2p3.py --test --model path_to_model_file -vis
  ```
   This will visualize the predictions in the results/ folder. You can also use the --num_samples parameter to specify how many predictions the model will output. If --num_samples > 3 the predictions will be clustered into 3 trajectories (using k-means), assigning a probability to each.

