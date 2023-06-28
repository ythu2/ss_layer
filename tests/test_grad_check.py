"""Implements unit test for checking backward passes."""
import unittest
import torch

from torch.autograd import gradcheck
from tests.create_toy_mesh import create_batched_toy_mesh
from surface_snapping.surface_snapping_e2e import SurfaceSnappingE2E

torch.manual_seed(0)

class TestBackwards(unittest.TestCase):
  def test_optimizer_backward_end_to_end(self):
    mesh = create_batched_toy_mesh()
    batched_verts = mesh.verts_padded()+1.321
    batched_verts.requires_grad = True
    batched_faces = mesh.faces_padded()
    batched_normals = mesh.faces_normals_padded() + 0.123
    batched_normals.requires_grad = True
    alpha = torch.ones(batched_verts.shape[0], 1, 1)*3.2127
    alpha.requires_grad = True

    solve = SurfaceSnappingE2E.apply
    gradcheck(solve, [batched_verts.double(),
                      batched_faces.double(),
                      batched_normals.double(),
                      alpha.double(),
                      1e-12,
                      1000000,
                      ])

if __name__ == '__main__':
  unittest.main()
