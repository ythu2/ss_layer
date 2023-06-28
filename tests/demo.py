import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from surface_snapping.surface_snapping_e2e import (
    snapping,
)
from surface_snapping.helpers import (
    normal_to_RGB,
)

device = 'cuda'

# Load the input image and the estimated surface normal
# The estimated surface normal is the result by [1]
# [1] Robust Learning Through Cross-Task Consistency, Zamir et al., CVPR 2020.
image = np.array(Image.open('data/image.jpg'))
normals = torch.Tensor(np.load('data/normal_map.npy')).to(device)

# Visualize normal
plt.imshow(normal_to_RGB(normals)[0].permute(1, 2, 0).cpu().numpy())
plt.savefig('surface_normal_estimate.png')

# Load the mesh
obj_data = load_obj('data/before_snapping.obj')
mesh_before_snapping = Meshes(
    [obj_data[0]], [obj_data[1].verts_idx]).to(device)


mesh_after_snapping = snapping(mesh_before_snapping, normals, alpha=0.5)
save_obj('output_0.5.obj', mesh_after_snapping.verts_list()
         [0], mesh_after_snapping.faces_list()[0])

mesh_after_snapping = snapping(mesh_before_snapping, normals, alpha=1.0)
save_obj('output_1.0.obj', mesh_after_snapping.verts_list()
         [0], mesh_after_snapping.faces_list()[0])

mesh_after_snapping = snapping(mesh_before_snapping, normals, alpha=2.0)
save_obj('output_2.0.obj', mesh_after_snapping.verts_list()
         [0], mesh_after_snapping.faces_list()[0])
