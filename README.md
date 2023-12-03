# dRIConv: Manufacturing process classification based on distance rotationally invariant convolutions
This is the github code for the paper "Manufacturing process classification based on distance rotationally invariant convolutions"
The environment for this code is old and as a result it is tough to generate. 
For convenience, you can directly utilize "env.yml" through the following code "conda env create -f env.yml" to generate the virtual environment.

The code for generating 3D files can be found in the following repository "https://github.com/ZhichaoWang970201/HKS-CNN".
You need to sample uniformly on the surface of the 3D model and utilize the sampled point clouds for classification.

I have created 3D models and sampling 1024 points in each model. 
They are saved in the following position onedrive path and you should put them into "dRIConv/data/modelnet40_ply_hdf5_2048".
"https://gtvault-my.sharepoint.com/:f:/g/personal/zwang945_gatech_edu/EhfNUOPsAjNGqO5Ma2bJrJ4Bo-Aa-9RV6ODqGk-HUFNesg?e=QSeaht"

Training:
`python3 train_val_cls.py`
