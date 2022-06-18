
# Data

- 15 joints
- 6,000 training mocap
- 800 test mocap
- 209 test mupots

datapoints
- input shape: (3, 15, 45)
- output shape: (3, 46, 45)
- (persons, frames, no_joints*3)

- `input_[:,1:15,:]-input_[:,:14,:]` = taking delta
- code at some locations implies that persons is shape[1], but it must be shape 0

# Setup Python Env outside Docker
- conda create -n mrt python=3.7.3 
- conda install -y pip numpy
- conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit==9.0 -c pytorch
- pip install torch-dct transforms3d matplotlib pygame pyopengl open3d==0.15.2 trimesh
- conda install numba
- conda install -c conda-forge jupyterlab

TODO:
- reread datasets & results section, find out which is done with which data