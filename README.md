# Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates

### [Project Page](http://www.computationalimaging.org/publications/deep-adaptive-lidar/) | [Paper](http://www.computationalimaging.org/wp-content/uploads/2020/03/deep-adaptive-lidar.pdf) | [Presentation (@31:45)](https://www.youtube.com/watch?v=9WAz9Y9gXgM&feature=youtu.be) ###

This repository contains the code associated with the paper Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates. Here, we propose an adaptive sampling scheme for LiDAR systems that demonstrates state-of-the-art performance for depth completion at low sampling rates. Our system is fully differentiable, allowing the sparse depth sampling and the depth inpainting components to be trained end-to-end with an upstream task.

## Usage
### Installation ###
An anaconda environment with all the dependences for this repository can be created using:
```
conda env create -f environment.yml
```

### Code Structure ###
The code is organized as follows:
* train_refinement.py & train_adaptive.py train the refinement (depth completion) network and the adaptive sampling network respectively.
* utils.py contains utility functions
* models directory contains the PyTorch models for the flow field prediction network, bilateral proxy network, monocular depth estimation network, and depth completion (refinement) network in their respective files.
  * densedepth.py contains the monocular depth estimation network
  * fast_bilateral_solver.py contains an implementation of the fast bilateral solver, which we fit a neural network to the output of
  * models_refinement.py contains a refinement model used in the NYU experiments
  * models_std.py contains a refinement model adapted from Self-Supervised Sparse-to-Dense which is used for refinement on the KITTI dataset.
  * parameter_prediction.py contains the differentiable sampling module
  * pytorch_prototyping(_orig).py contain implementations for common CNN building blocks, used for the bilateral solver proxy and for other parts of models built.
* helper_scripts directory contains scripts for processing and inpainting data
* dataloaders directory contains the data input/output for various datasets.
  * kitti.py contains the dataloader for the KITTI dataset
  * kitti_inpainted.py loads the inpainted KITTI dataset using the scripts in helper_scripts
  * nyu_v2.py contains the NYU-Depth-V2 data used in sparse-to-dense
  * nyu_v2_wonka.py contains NYU-Depth-V2 data used in the densedepth paper for training the monocular depth estimation network. We use this data for training out NYU-Depth-V2 model.
  * transforms*.py contain useful transforms for both datasets being loaded

### Training ###
Select a GPU to train on (either setting visible devices or manually passing in GPU number as an argument) and train the refinement and adaptive sampling models using:

```
python train_refinement.py
python train_adaptive.py
```

Dataset directory locations along with other training parameters and logging directories must be passed in using the command line arguments. The two dataset arguments are 'nyu_v2' or 'kitti'. Pre-trained models can be passed in by their directory location.

### Testing ###

Run the training scripts with the following options in order to evaluate models. Since we compare on the KITTI validation dataset, the following scripts can be run:

```
python train_refinement.py --eval_only /path/to/saved/model
python train_adaptive.py --eval_only /path/to/saved/model
```

### Data ####

The data for NYU-Depth-V2 can be found [here](https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo), the data does not need to be extracted to be loaded into the dataloader in nyu_v2_wonka.py.

The scripts for preprocessing the KITTI data (inpainting) is located in the helper_scripts subdirectory. Change the directory output locations desired in the file and run:
```
python inpaint.py
```
The sparse KITTI data used can be obtained from the annotated data available [here](http://www.cvlibs.net/datasets/kitti/).

### Pre-trained Models ###

Pre-trained models can be found [here](https://drive.google.com/open?id=1ovN05164zJsW1Nk2jJQJxI3XLarJyh5y). Models are given for both both KITTI and NYU-Depth-V2 in all classes. For refinement, we use a different model for 50 samples and for 200 samples on NYU-Depth-V2. 

## Misc
### Citation ###
If find our work useful in your research, please cite:
```
@article{Bergman:2020:DeepLiDAR,
author={Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
journal={Proc. IEEE ICCP},
title={{Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates}},
year={2020},
}
```

### Contact ###
If you have any questions, comments, or need assistance in navigating the repository, please contact Alex Bergman at awb@stanford.edu. 
