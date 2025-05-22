# SAC-based Mobile Robot Navigation

This repository contains Python code for training and testing
a SAC-based end-to-end autonomous navigation agent in Isaac Sim using a Scout Mini robot.

## Files
- `Train_SAC.py`: Main training script
- `Train_SAC_continue.py`: Resume training from checkpoint
- `custom_cnn.py`: Convolutional neural network for camera image encoding
- `ros_interface.py`: ROS2 interface for real-time interaction
- `end_to_end_nav_env.py`: Isaac Sim environment wrapper for SAC agent

## Note
URDF, USD map files, and Isaac Sim setup (robot model, simulation world) are assumed to be preconfigured in the user environment.
