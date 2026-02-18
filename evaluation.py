import torch
from physics import u3d_closed
from pathlib import Path

def create_xyz(x_range, y_range, z_range, nx=200, ny=200, nz=100):
    """
    Creates regularly spaced xyz pytorch tensor
    Args:
        x_range (tuple): (x0, x1) tuple of floats
        y_range (tuple): (y0, y1) tuple of floats
        z_range (tuple): (z0, z1) tuple of floats
        nx (int): number of nodes along the axis x
        ny (int): number of nodes along the axis y
        nz (int): number of nodes along the axis z
    Return:
        xyz (torch.Tensor): regularly spaced 3D Tensor
    """
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]
    assert z_range[0] <= z_range[1]

    x = torch.linspace(x_range[0], x_range[1], nx)  # Define x range
    y = torch.linspace(y_range[0], y_range[1], ny)  # Define y range
    z = torch.linspace(z_range[0], z_range[1], nz)  # Define z range

    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")  # Create 3D meshgrid
    xyz = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)  # Flatten and stack
    return xyz

def pinn_u3d_from_k_list(
    model, xz, k_vals, w_vals=None, assume_sorted=False, return_extras=False, chunk_size=None, detach=True, get_u_yy=False, kernel: str = 'cos'):
    r"""
    High-precision (float64) quadrature of the 3D potential on y=0:

      u_3D(x,z; y=0) = (1/pi) * ∫_0^{Kmax}  û(x,z;k) dk,
      where û is predicted by PINN via: model(xzk) with xzk = (x,z,log10k).

    If get_u_yy=True:
      - kernel='cos' : u_yy = -(1/pi) ∫ k^2 û dk
      - kernel='exp' : u_yy = +(1/pi) ∫ k^2 û dk

    Inputs:
      xz:      [N,2] or [B,N,2]
      k_vals:  [M] strictly positive
      w_vals:  [M] optional quadrature weights (else trapezoid over k_vals)
    """
    # ---- validate & cast ----
    if not isinstance(xz, torch.Tensor):
        raise TypeError("xz must be a torch.Tensor with shape [N,2] or [B,N,2].")
    device = xz.device
    xz = xz.double()

    k_vals = torch.as_tensor(k_vals, device=device, dtype=torch.float64)
    if k_vals.numel() < 1:
        raise ValueError("k_vals is empty.")
    if torch.any(k_vals <= 0):
        raise ValueError("k_vals must be strictly positive.")

    if not assume_sorted:
        k_vals, idx = torch.sort(k_vals)
        if w_vals is not None:
            w_vals = torch.as_tensor(w_vals, device=device, dtype=torch.float64)[idx]
    else:
        if w_vals is not None:
            w_vals = torch.as_tensor(w_vals, device=device, dtype=torch.float64)

    # ---- shape to [B,N,2] ----
    if xz.dim() == 2:
        xz = xz.unsqueeze(0)  # [1,N,2]
        squeeze_batch = True
    elif xz.dim() == 3:
        squeeze_batch = False
    else:
        raise ValueError("xz must be [N,2] or [B,N,2].")
    B, N, _ = xz.shape
    M = k_vals.numel()

    # ---- model evaluation over k blocks ----
    def _eval_block(k_block: torch.Tensor):
        m = k_block.numel()
        # repeat xz per k in the block: [B*m, N, 2]
        xz_rep = xz.repeat_interleave(m, dim=0)
        # build log10k column per sample: [B*m, N, 1]
        logk_block = torch.log10(k_block).view(1, m).repeat(B, 1)      # [B,m]
        logk_block = logk_block.reshape(B*m, 1).repeat(1, N).unsqueeze(-1)  # [B*m,N,1]
        # concat to (x,z,log10k): [B*m, N, 3]
        xzk_block = torch.cat([xz_rep, logk_block], dim=-1)
        # cast in/out
        uh = model(xzk_block.float())                   # [B*m,N,1]
        return uh.double().view(B, m, N, 1).squeeze(-1) # [B,m,N]

    if chunk_size is None or M <= chunk_size:
        uhat_BMN = _eval_block(k_vals)
    else:
        blocks = []
        for s in range(0, M, chunk_size):
            blocks.append(_eval_block(k_vals[s:s+chunk_size]))
        uhat_BMN = torch.cat(blocks, dim=1)  # [B,M,N]

    # ---- integrate over k ----
    def _integrate_over_k(values_BMN, weights=None):
        if weights is not None:
            w = weights.view(1, M, 1)
            out_BN = torch.sum(values_BMN * w, dim=1)
        else:
            if M < 2:
                raise ValueError("Need >=2 k nodes for trapezoidal integration (or provide weights).")
            dk = (k_vals[1:] - k_vals[:-1]).view(1, M-1, 1)
            f0 = values_BMN[:, :-1, :]
            f1 = values_BMN[:,  1:, :]
            out_BN = torch.sum(0.5 * (f0 + f1) * dk, dim=1)
        return out_BN  # [B,N]

    integral_BN = _integrate_over_k(uhat_BMN, w_vals)
    u_pred = (1.0 / torch.pi) * integral_BN.unsqueeze(-1)  # [B,N,1]

    # ---- optional u_yy ----
    u_yy_pred = None
    if get_u_yy:
        k2 = (k_vals**2).view(1, M, 1)
        sign = -1.0 if kernel == 'cos' else +1.0
        integral_yy_BN = _integrate_over_k(sign * uhat_BMN * k2, w_vals)
        u_yy_pred = (1.0 / torch.pi) * integral_yy_BN.unsqueeze(-1)  # [B,N,1]

    # ---- detach & cast back ----
    if detach:
        u_pred = u_pred.detach()
        uhat_BMN = uhat_BMN.detach()
        if u_yy_pred is not None:
            u_yy_pred = u_yy_pred.detach()

    if squeeze_batch:
        u_pred = u_pred.squeeze(0)
        if u_yy_pred is not None:
            u_yy_pred = u_yy_pred.squeeze(0)

    u_pred = u_pred.float()
    if u_yy_pred is not None:
        u_yy_pred = u_yy_pred.float()

    if return_extras:
        extras = (k_vals.float(), (uhat_BMN.squeeze(0) if squeeze_batch else uhat_BMN).float())
        return (u_pred, u_yy_pred, *extras) if get_u_yy else (u_pred, *extras)

    return (u_pred, u_yy_pred) if get_u_yy else u_pred

def get_components_secondary_u(u, xz):
    """
    Get the derivatives of the secondary 3D potential u(x,y,z).
    u_y=0. It does not calculate u_yy!!!
    Args:
        u (torch.tensor): [B, 1]
        xz (torch.tensor):   [B, 2], coordinates (x,z)
    Return:
        tuple of torch.tensor: (u, u_x, u_z, u_xx, u_zz) [B, 1]
    """
    assert u.dim() == 2 and u.size(-1) == 1, f"u must have shape [B, 1], got {u.shape}"
    assert xz.dim() == 2 and xz.size(-1) == 2, f"xyz must have shape [B, 2], got {xz.shape}"
    assert u.shape[0] == xz.shape[0], f"batch must match: u {u.shape}, xz {xz.shape}"
    assert u.dtype == xz.dtype, "u and xz must have the same dtype"
    assert u.device == xz.device, "u and xz must be on the same device"

    grad_u = torch.autograd.grad(u, xz, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[..., 0:1]  # ∂u/∂x
    u_z = grad_u[..., 1:2]  # ∂u/∂z

    # Second derivatives
    grad_grad_u_x = torch.autograd.grad(u_x, xz, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    grad_grad_u_z = torch.autograd.grad(u_z, xz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    # Extract the components
    u_xx = grad_grad_u_x[..., 0:1]  # ∂²u/∂x²
    u_zz = grad_grad_u_z[..., 1:2]  # ∂²u/∂z²
    return u, u_x, u_z, u_xx, u_zz

def pde_resid_u3D(xs, xz, u_components, u_yy, sigma_components, sigma0):
    """
    Calculates PDE residual for the Poisson equation with current point source. The singularity removal is done with decomposing potential into primary and secondary.
    ∇(σ₀ ∇u₁) + ∇(σ₁ ∇u₀) + ∇(σ₁ ∇u₁)= 0

    Args:
        xs (float): x-coordinate of source point
        xz (torch.tensor): [B, 2] containing (x,z) coordinates
        u_components (tuple of tensors): (u, u_x, u_z, u_xx, u_zz) [B, 1]
        u_yy (torch.tensor): [B, 1]
        sigma_components (tuple of tensors): (sigma, sigma_x, sigma_z) [B, 1]
        sigma0 (float)
    Return:
        torch.tensor: [B, 1] PDE residual for 3D u potential
    """
    assert xz.ndim == 2 and xz.shape[-1] == 2, f"xz must be [B, 2], got {tuple(xz.shape)}"

    # Add y=0 channel
    zeros = torch.zeros_like(xz[..., :1])  # [B,1]
    xyz = torch.cat((xz[..., :1], zeros, xz[..., 1:2]), dim=-1)
    #xyz = torch.cat((xz, torch.zeros_like(xz[..., :1])), dim=-1) # adds y=0 channel

    # Get the primary potential and its derivatives
    u0, u0_x, u0_y, u0_z, u0_xx, u0_yy, u0_zz = u3d_closed(xyz, xs, sigma0, get_components=True)
    lap_u0 = u0_xx + u0_yy + u0_zz # laplacian

    # get the same for the secondary potential
    u, u_x, u_z, u_xx, u_zz = u_components
    lap_u = u_xx + u_yy + u_zz

    # Get components of the secondary (heterogeneous) sigma
    sigma, sigma_x, sigma_z = sigma_components

    # Get PDE residual
    # ∇(σ₀ ∇u₁) + ∇(σ₁ ∇u₀) + ∇(σ₁ ∇u₁)= 0
    # [σ₀∇²u₁] + [∇σ₁∇u₀ + σ₁∇²u₀] + [∇σ₁∇u₁ + σ₁∇²u₁]
    term1 = sigma0 * lap_u  # [σ₀∇²u₁]
    term2 = (sigma_x * u0_x + sigma_z * u0_z) + (sigma * lap_u0)  # [∇σ₁∇u₀ + σ₁∇²u₀]
    term3 = (sigma_x * u_x + sigma_z * u_z) + (sigma * lap_u)  # [∇σ₁∇u₁ + σ₁∇²u₁]

    pde_res = term1 + term2 + term3
    return pde_res
