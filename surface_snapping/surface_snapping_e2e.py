"""Implements Vertex Update."""

import torch
import pytorch3d
from surface_snapping.cg_batch import cg_batch
from surface_snapping.helpers import (
  structure, 
  flatten, 
  _get_ND, 
  sparse_dense_mul, 
  sparse_dense_mul_prod,
  compute_average_normals_padded,
)

DEBUG = False
class SurfaceSnappingE2E(torch.autograd.Function):
  @staticmethod
  def forward(ctx, batched_verts, batched_faces, batched_normals,
              kappa, rtol, max_iter):
    """Performs Surface Snapping. 
    Note: alpha in the paper is 1/kappa in code,
          i.e., kappa is multiplied on the vertex consistent term.
    """
    batch_size, max_num_verts, _ = batched_verts.shape
    # Get components of A.
    ND, D0 = _get_ND(batched_faces, max_num_verts, batched_normals)
    
    # Compute b
    if kappa is not None:
      b = kappa*flatten(batched_verts, batch_size)

    # Defined iteration to conjugate gradient.
    def batched_RTRp(p):
      q_bar_top = ND.bmm(p)
      ret = ND.transpose(1, 2).bmm(q_bar_top)
      if kappa is not None:
        ret += kappa*p
      return ret

    ctx.max_iter = max_iter
    ctx.rtol = rtol
    x, _ = cg_batch(batched_RTRp, b, rtol=rtol, maxiter=max_iter)

    ctx.save_for_backward(ND, D0, b, kappa, x, batched_verts,
                          batched_normals)
    ctx.normals_requires_grad = batched_normals.requires_grad
    if DEBUG:
      print(x.min())
      print(x.max())
      print(x.mean())
      print(batched_verts.min())
      print(batched_verts.max())
      print(batched_verts.mean())
    return x

  @staticmethod
  def backward(ctx, grad):
    ND, D0, b, kappa, x, batched_verts, batched_normals = ctx.saved_tensors
    gradkappa = None
    grad_batched_normals = None  # TODO: Implement.
    batch_size = x.shape[0]

    # Defined iteration to conjugate gradient for backward.
    def batched_RTRp(p):
      q_bar_top = ND.bmm(p)
      ret = ND.transpose(1, 2).bmm(q_bar_top)
      if kappa is not None:
        ret += kappa*p
      return ret

    # Compute dL/db
    gradb, _ = cg_batch(batched_RTRp, grad, rtol=ctx.rtol,
                        maxiter=ctx.max_iter)

    # Compute dL/dv
    if batched_verts.requires_grad:
      # Chian rule: dL/dv = dL/db*db/dv = dL/db*kappa
      grad_batched_verts = structure(gradb, batch_size)*kappa

    # Compute dL/dkappa
    if kappa.requires_grad:
      # dL/dkappa through b
      term1 = (gradb*flatten(batched_verts, batch_size)).squeeze().sum(-1)
      # dL/dkappa through A
      term2 = (-gradb*x).sum(-1).sum(-1)
      gradkappa = (term1 + term2).unsqueeze(-1).unsqueeze(-1)
      
    # Compute dL/dN
    if batched_normals.requires_grad:
      batch_size, m, n = ND.shape
      # Split into two low dim bmm.
      tmp1 = ND.bmm(-gradb)
      tmp2 = ND.bmm(x)
      Nx = sparse_dense_mul_prod(D0, tmp1, tmp2, x.transpose(1,2), gradb.transpose(1,2), offset=0)
      Nx = torch.sparse.sum(Nx,-1).unsqueeze(1)
      Ny = sparse_dense_mul_prod(D0, tmp1, tmp2, x.transpose(1,2), gradb.transpose(1,2), offset=1)
      Ny = torch.sparse.sum(Ny, -1).unsqueeze(1)
      Nz = sparse_dense_mul_prod(D0, tmp1, tmp2, x.transpose(1,2), gradb.transpose(1,2), offset=2)
      Nz = torch.sparse.sum(Nz, -1).unsqueeze(1)
      Nxyz = torch.cat([Nx, Ny, Nz], 1).transpose(1, 2).to_dense()
      Nxyz_m = Nxyz.view(batch_size, -1, 3, 3).sum(2)
      grad_batched_normals = Nxyz_m
    return grad_batched_verts, None, grad_batched_normals, gradkappa, None, None,

def fit_surface_snapping_e2e(batched_verts, batched_faces, batched_normals,
                             kappa=1, rtol=1e-8, max_iter=1000):
  # Wrapper function to use Surface Snapping.
  assert kappa < 100 and kappa >= 0
  batch_size, max_num_verts, _ = batched_verts.shape
  device = batched_verts.device

  if kappa is not None:
    if not torch.is_tensor(kappa):
      kappa = torch.ones(batch_size, 1, 1, device=device,
                         requires_grad=False)*kappa
    else:
      assert len(kappa.shape) == 3  # Batchx1x1
  x_final = SurfaceSnappingE2E.apply(batched_verts, batched_faces, batched_normals,
                                     kappa, rtol, max_iter)
  x_final = x_final.reshape(batch_size, 3, -1).transpose(1, 2)
  return x_final


def snapping(mesh, normals, alpha=1.0):
  """
  Params:
      mesh: input mesh, pytorch3d.structures.Meshes, N meshes
      normals: Nx3xHxW, normal map
      alpha: 
      Note: alpha in the paper is 1/kappa in code,
          i.e., kappa is multiplied on the vertex consistent term.
  Returns:
      mesh_updated: snapped mesh, pytorch3d.structures.Meshes, N meshes
  """
  # transform from prediction coordinate to pytorch NDC
  verts = mesh.verts_list()
  verts = [v.clone() for v in verts]
  for v in verts:
    v[:, 0] = -v[:, 0] 
    v[:, 1] = -v[:, 1] 
    v[:, 2] = v[:, 2]/2.0 + 0.5 # original in [-1,1], make it greater than 0 such that it can be rendered
  mesh_aligned = pytorch3d.structures.Meshes(verts=verts, faces=mesh.faces_list())
  
  avg_normal, visibility = compute_average_normals_padded(mesh_aligned, normals)

  verts = mesh_aligned.verts_padded() 
  faces = mesh_aligned.faces_padded() # faces: bsxnfx3

  verts_updated = fit_surface_snapping_e2e(verts.clone(), 
                                            faces, 
                                            avg_normal, 
                                            kappa=1./alpha)

  # convert from packed to list
  n_verts = mesh_aligned.num_verts_per_mesh()
  verts_updated_list = []
  for i in range(len(mesh)):
      verts_updated_list.append(verts_updated[i,:n_verts[i]])

  # transform back to prediction coordinate
  for v in verts_updated_list:
    v[:,0] *= -1
    v[:,1] *= -1
    v[:,2] = (v[:,2]-0.5)*2
  
  mesh_updated = pytorch3d.structures.Meshes(verts=verts_updated_list, faces=mesh_aligned.faces_list())
  return mesh_updated
