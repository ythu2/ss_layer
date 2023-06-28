# Surface Snapping Optimization Layer for Single Image Object Shape Reconstruction

### ICML 2023 ([PDF](https://openreview.net/pdf?id=C8ijRC4ZvS)))
[Yuan-Ting Hu](https://sites.google.com/view/yuantinghu),
[Alexander G. Schwing](http://www.alexander-schwing.de/),
[Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>1</sup><br>
University of Illinois at Urbana-Champaign <br/>
Purdue University<sup>1</sup><br/>

# Overview
This repository contains code for Surface Snapping Optimization Layer for Single Image Object Shape Reconstruction
accepted at ICML 2023.

If you used this code or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{hu-icml2023-surface,
  title = {Surface Snapping Optimization Layer for Single Image Object Shape Reconstruction},
  author = {Yuan-Ting Hu and Schwing, Alexander G and Yeh, Raymond A},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2023},
}
</pre>

## Setup Dependencies
To install the dependencies, run the following
```bash
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c conda-forge matplotlib
conda install pytorch3d -c pytorch3d
conda install conda-build
cd surface_snapping_code
conda develop .
```

## Run Tests
```bash
python -m unittest discover tests/
```

## Demo
To run the demo code, simply run

```bash
python tests/demo.py
```

The demo code runs the proposed surface snapping algorithm to refine the input mesh (`data/before_snapping.obj`) using the predicted surface normals (`data/normal_map.npy`). The resulting snapped mesh is stored in `output_[alpha].obj`.

We also provide a juypter-notebook version of this demo (`demo/surface_snapping_demo.ipynb`). To run the notebook, please follow the instructions below:
```bash
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
jupyter-notebook demo/surface_snapping_demo.ipynb
```
