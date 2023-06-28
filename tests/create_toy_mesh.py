import time
import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

def create_toy_mesh():
  p0 = [0,  0, 0]
  p1 = [-1, 1, 0]
  p2 = [-1, -1, 0]
  p3 = [-2, 0, 0]
  verts = torch.Tensor([p0, p1, p2, p3])
  faces = torch.Tensor([[0, 1, 2], [1, 2, 3]])  # ,[1,2,3]])
  print(verts)
  print(faces)
  verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
  device = 'cuda'
  textures = Textures(verts_rgb=verts_rgb.to(device))
  return Meshes(verts=[verts], faces=[faces], textures=textures)


def create_batched_toy_mesh():
  # Mesh 1
  verts1 = torch.Tensor([[0,  0, 0], [-1, 1, 0], [-1, -1, 0], [-2, 0, 0]]) #,[-2, 0, 0]]) # [-2,-1,0]])
  faces1 = torch.Tensor([[0, 1, 2], [1, 2, 3]]) #, [2,3,4]])
  # Mesh 2
  verts2 = torch.Tensor([[0,  0, 0], [0, -1, 1], [0, -1, -1]])
  faces2 = torch.Tensor([[0, 1, 2]])
  return Meshes(verts=[verts1, verts2], faces=[faces1, faces2])
