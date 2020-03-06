## Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates ##

This repository contains the code associated with the paper Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates. 

The script for training the refinement network is located in train_refinement.py
The script for training the adaptive sampling network, and tuning the refinement end-to-end is in train_adaptive.py

The data for NYU Depth v2 can be found at: https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo, the data does not need to be extracted to be loaded into the method.
The scripts for preprocessing the KITTI data (inpainting) is located in the helper_scripts subdirectory. This can be obtained from the annotated data available at http://www.cvlibs.net/datasets/kitti/

Pre-trained models can be found at:
https://drive.google.com/open?id=1ovN05164zJsW1Nk2jJQJxI3XLarJyh5y
