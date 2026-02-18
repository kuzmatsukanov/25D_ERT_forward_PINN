import torch

def _safe(x, eps=1e-12):
    return torch.clamp(x, min=eps)

def get_analytical_solution(x_s, xzk, sigma0, I=1.0, out_dtype=None, get_residual=False):
    r"""
    Analytical 2.5D solution in Fourier (y) space for a uniform half-space (σ = σ0), y=0 slice.

    PDE (per batch and wavenumber k):
        ∂_x(σ0 ∂_x φ0) + ∂_z(σ0 ∂_z φ0) − k^2 σ0 φ0 = −δ(x−x_s, z)

    Closed-form solution:
        φ0(x, z; k) = (I / (π σ0)) K0(|k| ρ),    ρ = sqrt((x−x_s)^2 + z^2)
    Internally evaluates all expressions in float64 for numerical stability, then casts results back to `out_dtype`

    Args:
        x_s (float): position of the electrode, Source x-location (z_s = 0).
        xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
        sigma0 (float): background electrical conductivity
        out_dtype (torch.dtype | None): output dtype; if None, use xzk.dtype
        get_residual (bool): if True, also return PDE residual
    Return:
        tuple of torch.Tensor (u, u_x, u_z, u_xx, u_zz, residual) each of shape [B, 1], in dtype=out_dtype.
    """
    if out_dtype is None:
        out_dtype = xzk.dtype

    device = xzk.device

    # --- cast to float64 for special functions & derivatives ---
    xzk64 = xzk.to(torch.float64)              # [B, 3]
    x64   = xzk64[..., 0]                      # [B]
    z64   = xzk64[..., 1]                      # [B]
    logk  = xzk64[..., 2]                      # [B]  (log10 k)
    k64   = torch.pow(10.0, logk)              # [B]
    kabs  = k64.abs()                          # [B] (redundant since k>0, but safe)

    B, _ = xzk.shape
    xs64 = torch.as_tensor(x_s, dtype=torch.float64, device=device)
    I64 = torch.as_tensor(I, dtype=torch.float64, device=device)
    sigma064 = torch.as_tensor(sigma0, dtype=torch.float64, device=device)

    # --- geometry ---
    dx = x64 - xs64                    # [B] via broadcast
    rho = torch.sqrt(dx * dx + z64 * z64)  # [B]
    rho = rho.clamp_min(1e-12)  # [B]

    a = (kabs * rho).clamp_min(1e-12)          # [B]

    # --- constants & Bessel functions ---
    #A = I64 / (2 * torch.pi * sigma064)    # scalar
    A = I64 / (torch.pi * sigma064)  # scalar
    K0 = torch.special.modified_bessel_k0(a)
    K1 = torch.special.modified_bessel_k1(a)

    # potential
    u = A * K0                         # [B]

    # first derivatives
    drdx = dx / rho                    # [B]
    drdz = z64 / rho                   # [B]
    f1 = -A * (kabs * K1)              # [B]
    u_x = f1 * drdx
    u_z = f1 * drdz

    # second derivatives
    rho3 = rho * rho * rho
    d2rdx2 = (z64 * z64) / rho3        # [B]
    d2rdz2 = (dx * dx) / rho3          # [B]
    f2 = A * (kabs * kabs) * (K0 + K1 / a)  # [B]
    u_xx = f2 * (drdx * drdx) + f1 * d2rdx2
    u_zz = f2 * (drdz * drdz) + f1 * d2rdz2

    # --- cast back & add trailing dim ---
    def to_out(t: torch.Tensor) -> torch.Tensor: # [...,] -> [..., 1]
        return t.to(dtype=out_dtype, device=device)[..., None]  # [B, 1]

    if get_residual:
        residual = u_xx + u_zz - (k64 * k64) * u  # [B]
        return to_out(u), to_out(u_x), to_out(u_z), to_out(u_xx), to_out(u_zz), to_out(residual)  # [B, 1]
    else:
        return to_out(u), to_out(u_x), to_out(u_z), to_out(u_xx), to_out(u_zz)  # [B, 1]

def uhat_k0(x, z, k, xs, sigma0, I=1.0):
    """2.5-D homogeneous Green's function in k-space."""
    rho = torch.sqrt((x - xs)**2 + z**2)
    a = torch.abs(k) * rho
    A = I / (torch.pi * sigma0)
    return A * torch.special.modified_bessel_k0(_safe(a))


def u3d_closed(xyz, xs, sigma0, I=1.0, get_components=False):
    r"""Closed-form 3D half-space potential (Neumann at z=0) for the point source at (x_s,0,0)
    $\nabla \cdot [\sigma(x,z) \nabla u(x,y,z)] = - I\cdot\delta(x - x_s,y, z)$
    u = I/(2\pi\sigma0 r_s)
    where r_s is the distance from the points source

    Args:
        xyz (torch.tensor): [B, 3] containing (x,y,z) coordinates
        xs (float): x-coordinate of source point
        sigma0 (float): conductivity parameter
        I (float): current intensity
        get_components (bool): return components of u0 as well

    Returns:
        torch.tensor: [B, 1] potential values
        u or (u, u_x, u_y, u_z, u_xx, u_yy, u_zz)
    """
    assert xyz.ndim == 2 and xyz.shape[-1] == 3, f"xyz must be [B, 3], got {tuple(xyz.shape)}"
    # Keep last-dim to preserve [B, 1] shapes
    x, y, z = xyz[..., 0:1], xyz[..., 1:2], xyz[..., 2:3]  # [B, 1]
    dx = x - xs

    R2 = dx ** 2 + y ** 2 + z ** 2
    R = R2.sqrt().clamp_min(1e-12) # [B, 1]

    # coeff = I/2\pi\sigma_0
    coeff = (torch.as_tensor(I, dtype=xyz.dtype, device=xyz.device) / (2.0 * torch.pi * torch.as_tensor(sigma0, dtype=xyz.dtype, device=xyz.device)))

    u = coeff / R  # [B,1]
    if not get_components:
        return u

    R3 = R2 * R
    R5 = R3 * R2

    # First derivatives
    u_x = -coeff * dx / R3
    u_y = -coeff * y  / R3
    u_z = -coeff * z  / R3

    # Second derivatives
    u_xx = coeff * (3.0 * dx**2 - R2) / R5
    u_yy = coeff * (3.0 * y**2  - R2) / R5
    u_zz = coeff * (3.0 * z**2  - R2) / R5
    return u, u_x, u_y, u_z, u_xx, u_yy, u_zz
