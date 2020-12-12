# hand-movement-tracking
![project] ![research] ![Python]

## Content

    1. Description
    2. Repository Structure
    3. Running the System
    4. Credit

## Description

System to identify and track hand movements in a video stream with a ego centric view point. The system uses a pre-trained segementation model trained on the [ego-centric dataset](http://vision.soic.indiana.edu/projects/egohands/). Video stream is converted to frames and preprocessed and passed into the pre-trained model for inference. The model output a segemented image which is then post processsed to identify contours to generate a box around it. Post processing is done to reduce the noise. 

## Repository Structure

The important files and directories of the repository is shown below

    ├── seg_hand_trained - contain the hand segemented model
    ├── Videos - Video folder that contains the input video
    ├── infer_segmentation.py - contain class for main inference
    ├── segmentation.py - main runn file, contain function to preprocess and post-process frames
    └── utils.py - contain utility functions

## Running the System

Cloning the git repository

    git clone https://github.com/iTharindu/hand-movement-tracking.git

Getting inside the repository

    cd hand-movement-tracking

Creating a virtual environment

    virtualenv --python=python3.6 env

    source env/bin/activate

If virtualenv is not installed then run `pip3 install virtualenv` to install virtualenv. 

Installing the dependencies

    pip3 install -r requirements.txt

To generate tracking video

    python segmentation.py

Output will be generated as `output.avi`

## Credit

We used a pre-trained model `Unet` model with the `efficientnetb2` backbone. This model with several other models are included in the [here](https://github.com/qubvel/segmentation_models)








