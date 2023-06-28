"""Implements helper functions."""

import torch
import pytorch3d
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes


def flatten(vv, batch_size):
  return vv.transpose(1, 2).contiguous().view(batch_size, -1, 1)


def structure(vv, batch_size):
  return vv.view(batch_size, 3, -1).transpose(1, 2)


def _get_ND(batched_faces, max_num_verts, batched_normals):
  diff_matrix_diag = construct_diff_matrix_fast_sparse(
      batched_faces, max_num_verts, batched_normals.dtype)
  normal_matrix_diag = construct_normal_matrix_fast(
      batched_normals).unsqueeze(-1)
  o1 = sparse_dense_mul(
      diff_matrix_diag, normal_matrix_diag[:, 0].expand(-1, -1, diff_matrix_diag.shape[-1]))
  o2 = sparse_dense_mul(
      diff_matrix_diag, normal_matrix_diag[:, 1].expand(-1, -1, diff_matrix_diag.shape[-1]))
  o3 = sparse_dense_mul(
      diff_matrix_diag, normal_matrix_diag[:, 2].expand(-1, -1, diff_matrix_diag.shape[-1]))
  ND = torch.cat([o1, o2, o3], -1)
  return ND, diff_matrix_diag


def sparse_dense_mul(s, d):
  """Sparse dense element-wise mul."""
  i = s._indices()
  v = s._values()
  # get values from relevant entries of dense matrix
  dv = d[i[0, :], i[1, :], i[2, :]]
  return torch.sparse.FloatTensor(i, v * dv, s.size())


def construct_diff_matrix_fast_sparse(batched_faces, max_num_verts, dtype=torch.float):
  """Fast Method to create a Sparse D Matrix."""
  device = batched_faces.device
  batch_size, max_num_faces, _ = batched_faces.shape
  o3_bb = batched_faces.reshape(-1)
  mo3_bb = batched_faces[:, :, [1, 2, 0]].reshape(-1)
  select_idx = o3_bb != -1
  o1 = torch.arange(
      batch_size, device=device).repeat_interleave(max_num_faces*3)
  o2 = torch.arange(max_num_faces*3, device=device).repeat(batch_size)

  o1 = o1[select_idx].unsqueeze(-1)
  o2 = o2[select_idx].unsqueeze(-1)
  o3_bb = o3_bb[select_idx].unsqueeze(-1)
  mo3_bb = mo3_bb[select_idx].unsqueeze(-1)

  tmp = torch.cat([torch.cat([o1, o2, o3_bb], -1),
                   torch.cat([o1, o2, mo3_bb], -1)], 0)
  one_m_one = torch.ones(tmp.shape[0], device=device)
  one_m_one[tmp.shape[0]//2:] = -1
  d_diff = torch.sparse_coo_tensor(tmp.transpose(0, 1), one_m_one,
                                   [batch_size, max_num_faces*3, max_num_verts],
                                   device=device, dtype=dtype)
  return d_diff


def construct_normal_matrix_fast(batched_normals):
  """Fast Method to creating N with only the diagonals"""
  device = batched_normals.device
  batch_size, max_num_faces, _ = batched_normals.shape
  batched_normals_rep = torch.cat(
      [batched_normals]*3, -1).reshape(batch_size, -1, 3).transpose(1, 2)
  return batched_normals_rep


def sparse_dense_mul_prod(s, d1, d2, d3, d4, offset):
  """Sparse dense element-wise mul with a lookup."""
  i = s._indices()
  v = s._values()
  # get values from relevant entries of dense matrix
  ww = d4.shape[-1]//3
  dv1 = d1[i[0, :], i[1, :], 0]
  dv2 = d2[i[0, :], i[1, :], 0]
  dv3 = d3[i[0, :], 0, i[2, :]+offset*ww]
  dv4 = d4[i[0, :], 0, i[2, :]+offset*ww]
  out = v * (dv1*dv3 - dv2*dv4)
  return torch.sparse.FloatTensor(i, out, s.size())


def normal_to_RGB(normals):
  """convert normal vectore to rgb for visualization"""
  #normal_maps: bsx3xnxn, torch tensor
  RGB = normals.clone()
  RGB[:,:2] = -(normals[:,:2]-1.0)*127.5
  RGB[:,2] = (normals[:,2]+1.0)*127.5
  return RGB/255.0


def compute_average_normals_padded(mesh, normals, perspective_correct=True):
  """
  Params:
      mesh: pytorch3d.structures.Meshes, N meshes
      normals: Nx3xHxW, normal map
  Returns:
      avg_normal: bsxnfx3, nf is the max number of faces among the input N meshes. nf = mesh.faces_padded().shape[1]
                  the average normal of th i-th face in the normal map.
      visibility: bsxnF, whether the i-th face is visible or not. 0 -> invisible, otherwise visible.
  """
  bs, nf, _ = mesh.faces_padded().shape

  pix2face, _, _, _ =  rasterize_meshes(mesh, image_size=256, blur_radius=0.0, faces_per_pixel=1, perspective_correct=perspective_correct)

  p2f = pix2face.clone().contiguous() # indices in faces_packed()
  idx = (p2f != -1)

  # convert indices to faces_padded()
  offsets = mesh.mesh_to_faces_packed_first_idx()
  p2f = p2f - offsets.view(-1,1,1,1)
  offsets = torch.arange(0, bs*nf, nf).to(p2f.device)
  p2f = p2f + offsets.view(-1,1,1,1)

  # select visible faces
  p2f = p2f.reshape(-1)
  p2f = p2f[idx.view(-1)]

  # select normals of visible faces
  n = normals.clone().permute(0,2,3,1).reshape(-1,3)
  n = n[idx.view(-1),:]

  agg_normal = torch.zeros((bs*nf, 3)).to(p2f.device)
  agg_normal.index_add_(0, p2f, n) 
  cnt = torch.bincount(p2f, minlength=bs*nf)
  visibility = cnt.clone()
  cnt[cnt==0] = 1
  avg_normal = agg_normal / cnt.view(-1,1)

  avg_normal = avg_normal.reshape(bs, nf, 3)
  visibility = visibility.reshape(bs, nf)

  return avg_normal.detach(), visibility.detach()

