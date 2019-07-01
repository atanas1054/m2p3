# imotion-m3p
  This repository contains the project "Egocentric Multi-pedestrian Path Prediction (M3P) in Traffic Scenarios" which is being developed in iMotion Germany by Atanas Poibrenski

# Dependencies
  
  The code was tested on Ubuntu 16.04,Python 3 and a GTX 1080 gpu . The following dependencies are needed:
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
  The dependencies can be installed using "pip install"


# Milestone 1 (M1)

  We demonstrate Milestone 1 by showing the bounding box and unique ID’s (of the pedestrians) together with the optical flow of the scene given the input images. The code uses the implementation of [1], [2] and [3]. In order to run the demo, the following steps should be taken:
  
  1. Download the extract the whole project
  2. Download the mask-RCNN weights from here https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
  3. Put the mask_rcnn_coco.h5 file in ./HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn/Mask_RCNN
  4. Download the optical flow model pwcnet-lg-6-2-multisteps-chairsthingsmix-20190314T111850Z-001.zip from https://drive.google.com/drive/folders/1iRJ8SFF6fyoICShRVWuzr192m_DgzSYp
  5. Extract pwcnet-lg-6-2-multisteps-chairsthingsmix in ./OpticalFlow/tpflow/tfoptflow-master/tfoptflow/models
  6. Navigate to ./HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn and run 
     
  ```
   python demo.py demo_video.mp4
  ```
  There are already two sample videos showing the results.

# Milestone 2 (M2)
  
  We demonstrate Milestone 2 by showing how to train/test a model on the JAAD dataset.

# Milestone 3 (M3) 
  
  N/A

# Milestone 4 (M4)

 End of July 2019

# References

[1] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2017.

[2] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in IEEE International Conference on Image Processing (ICIP), 2016.

[3] Deqing Sun et al. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume," (CVPR 2018)
