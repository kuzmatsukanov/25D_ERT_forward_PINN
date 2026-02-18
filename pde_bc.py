import torch


def get_source_correction_term(xzk, x_s, sigma1, sigma0, width, I=1.0):
    """
    Calculates the 'Missing Energy' correction term to add to the PDE residual.

    Formula: Correction = - (sigma1 / sigma0) * (I/2) * Delta_approx
    Where Delta_approx is a 2D Gaussian: (1 / (pi * w^2)) * exp(-r^2 / w^2)

    Args:
        xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
        x_s (float): position of the electrode, Source x-location (z_s = 0).
        sigma1 (torch.Tensor): The anomaly conductivity [B, 1] at coordinates (x,z).
        sigma0 (float): Background conductivity (scalar).
        I (float): Source current amplitude (scalar).
        width (float): The 'spread' of the Gaussian approximation (meters).

    Returns:
        torch.Tensor: The correction term [B, 1] to ADD to the PDE residual.
    """
    assert xzk.shape[0] == sigma1.shape[0]

    # --- enforce float64 ---
    xzk    = xzk.to(dtype=torch.float64)
    sigma1 = sigma1.to(dtype=torch.float64)

    device = xzk.device
    dtype  = torch.float64

    # 1. Safety Constants
    EPS = 1e-12  # Tiny number to prevent division by zero

    # 2. Validate Inputs
    # Ensure width is not zero or negative
    assert width > EPS, f"Width must be positive and non-zero. Got {width}"

    # Ensure sigma0 is not zero (physics breakdown if background is perfect insulator)
    assert abs(sigma0) > EPS, "sigma0 cannot be zero."

    # 2. Extract Coordinates (assuming x is col 0, z is col 1)
    # xzk shape is [B, 3]
    x = xzk[:, 0:1]
    z = xzk[:, 1:2]

    # 3. Calculate Squared Distance (r^2)
    # Since source is at (x_s,0)
    r_squared = (x - x_s) ** 2 + z ** 2

    # 3. Calculate Gaussian Factors
    # Factor A = 1 / (pi * w^2)
    # We use numpy for Pi, ensuring it's cast to the tensor's dtype/device
    pi_tensor = torch.tensor(torch.pi, device=device, dtype=dtype)
    width_tensor = torch.tensor(width, device=device, dtype=dtype)
    width_sq = width_tensor ** 2

    # 5. Gaussian Normalization Factor A = 1 / (pi * w^2)
    normalization = 1.0 / (pi_tensor * width_sq + EPS)

    # 4. Calculate Gaussian Exponent
    # exp(-r^2 / w^2)
    gaussian_shape = torch.exp(-r_squared / (width_sq + EPS))

    # 5. Assemble Delta Approximation
    # delta_approx [B, 1]
    delta_approx = normalization * gaussian_shape

    # 6. Calculate the Ratio
    # ratio = sigma1 / sigma0
    # sigma1 is [B, 1], sigma0 is scalar. Result is [B, 1]
    conductivity_ratio = sigma1 / sigma0

    # 7. Final Correction Term
    # Term = - (Ratio) * I * Delta
    # We cast 'I' to a tensor if it isn't one, to match device
    source_correction = -1.0 * conductivity_ratio * (4*I/2) * delta_approx
    return source_correction

class PdeBc:
    """Definition of PDE, BC residuals for ERT problem"""
    def __init__(self):
        pass

    def pde(self, u, u0_components, sigma_components, xzk, x_s, sigma0):
        """
        PDE residual for the secondary potential φ1 with singularity removal:
            ∇(σ₀∇φ₁) + ∇(σ₁∇φ₀) + ∇(σ₁∇φ₁) − k²(σ₀φ₁ + σ₁φ₀ + σ₁φ₁) = 0   [A/m^3]

        Args:
            u (torch.Tensor): of shape [B, 1] φ₁(x, z, k) solution, the predicted electric potential field (secondary potential)
            u0_components (iterable of torch.Tensors): φ₀(u0, u0_x, u0_z, u0_xx, u0_zz) of shapes [B, 1], the analytically computed primary potential and its derivatives
            sigma_components (iterable of torch.Tensors): (sigma, sigma_x, sigma_z) of shapes [B, 1], σ₁ heterogeneity
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
            x_s (float): position of the electrode, Source x-location (z_s = 0)
            sigma0 (float): background σ₀
        Returns:
            pde_res (torch.Tensor): of shape [B, 1] PDE residual.
        """
        # Basic shape checks
        assert u.dim() == 2 and u.size(-1) == 1, "u must be [B,1]"
        assert xzk.dim() == 2 and xzk.size(-1) == 3, "xzk must be [B,3]"
        assert xzk.requires_grad, "xz must require grad for autograd"

        # Unpack components
        u0, u0_x, u0_z, u0_xx, u0_zz = u0_components
        sigma, sigma_x, sigma_z = sigma_components

        # Device / dtype / finite checks (exclude x_s — it's a float)
        for t in (u, xzk, u0, u0_x, u0_z, u0_xx, u0_zz, sigma, sigma_x, sigma_z):
            assert t.device == u.device, "All tensors must be on the same device"
            #assert t.dtype == u.dtype, "All tensors must share dtype"
            assert torch.isfinite(t).all(), "Inputs contain NaN/Inf"

        # First derivatives of u wrt (x,z)
        grad_u = torch.autograd.grad(u, xzk, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[..., 0:1]  # ∂u/∂x [B,1]
        u_z = grad_u[..., 1:1 + 1]  # ∂u/∂z [B,1]

        # Second derivatives of u
        grad_grad_u_x = torch.autograd.grad(u_x, xzk, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        grad_grad_u_z = torch.autograd.grad(u_z, xzk, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        u_xx = grad_grad_u_x[..., 0:1]  # ∂²u/∂x² [B,1]
        u_zz = grad_grad_u_z[..., 1:1 + 1]  # ∂²u/∂z² [B,1]

        # Laplacians
        lap_u = u_xx + u_zz
        lap_u0 = u0_xx + u0_zz

        # PDE terms
        term1 = sigma0 * lap_u  # σ₀∇²φ₁
        term2 = (sigma_x * u0_x + sigma_z * u0_z) + (sigma * lap_u0)  # ∇σ₁·∇φ₀ + σ₁∇²φ₀
        term3 = (sigma_x * u_x + sigma_z * u_z) + (sigma * lap_u)  # ∇σ₁·∇φ₁ + σ₁∇²φ₁

        # k²(σ₀φ₁ + σ₁φ₀ + σ₁φ₁); k broadcasts from [B,1,1] over N
        logk = xzk[..., 2:3]
        k = (10.0 ** logk)
        k2 = k * k
        term4 = k2 * (sigma0 * u + sigma * u0 + sigma * u)

        source_correction = get_source_correction_term(xzk=xzk, x_s=x_s, sigma1=sigma, sigma0=sigma0, width=0.02)

        pde_res = term1 + term2 + term3 - term4 + source_correction
        return pde_res

    def bc_no_flow(self, u, sigma, sigma0, xzk):
        r"""
        Enforces the no-flow (Neumann) boundary condition on the surface:
        (σ₀ + σ₁)∂φ₁/∂z = 0     [A/m^2]
        Args:
            u (torch.Tensor): of shape [B, 1], predicted secondary electric potential field φ₁(x, k, z).
            sigma (torch.Tensor): of shape [B, 1], sigma(x, z).
            sigma0 (float): background electrical conductivity.
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
        Returns:
            (sigma0 + sigma) * ∂u/∂z (torch.Tensor): of shape [B, 1], boundary residual enforcing no-flow condition.
        """
        assert u.ndim == 2 and u.shape[-1] == 1, f"u must be [B,1], got {u.shape}"
        assert sigma.ndim == 2 and sigma.shape[-1] == 1, f"sigma must be [B,1], got {sigma.shape}"
        assert xzk.ndim == 2 and xzk.shape[-1] == 3, f"xz must be [B,3], got {xzk.shape}"
        assert u.shape[0] == xzk.shape[0], "Batch/point dims of u and xzk must match"
        assert sigma.shape[0] == xzk.shape[0], "Batch/point dims of sigma and xzk must match"
        assert xzk.requires_grad, "xzk must require grad for autograd"

        # device/dtype/finite checks
        for t in (u, sigma, xzk):
            assert t.device == u.device, "All tensors must be on the same device"
            assert t.dtype == u.dtype, "All tensors must share dtype"
            assert torch.isfinite(t).all(), "Inputs contain NaN/Inf"

        # ∂u/∂z via autograd; note: z is index 1 in (x,z)
        grad_u = torch.autograd.grad(u, xzk, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = grad_u[..., 1:2]
        return (sigma0 + sigma) * u_z

    def bc_robin(self, u, u0_components, xzk, x_s, bc_type):
        r"""
        Enforces the Robin absorbing BC on bottom and side boundaries in 2.5D (Dey, Morrison, 1979, p. 113):
        n⋅∇φ₁ + α⋅φ₁ = 0,  where  α = k cos(r^n) ⋅ K1(k⋅r) / K0(k⋅r),  r = sqrt((x - xs)^2 + z^2)
        units: [V/m]
        Args:
            u (torch.Tensor): of shape [B, 1], predicted electric potential field φ₁(x, k, z)
            u0_components (iterable of torch.Tensors): (φ₀, φ₀_x) or (φ₀, φ₀_z) of shapes [B, 1],
                the analytically computed primary potential and its derivatives. It is better be in float64
            x_s (float): position of the electrode, Source x-location (z_s = 0).
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
            bc_type (str): 'bottom', 'left', or 'right'
        """
        assert bc_type in ("bottom", "left", "right"), f"Unsupported BC type: {bc_type}. Use 'bottom', 'left', or 'right'."
        B, D = xzk.shape
        assert D == 3, f"xkz must have shape [B,3], got {xzk.shape}"
        assert u.shape == (B, 1), f"u must have shape [B,1], got {u.shape}"
        for idx, comp in enumerate(u0_components):
            assert isinstance(comp, torch.Tensor), f"u0_components[{idx}] must be a tensor"
            assert comp.shape == (B, 1), f"u0_components[{idx}] must have shape [B,1], got {comp.shape}"

        # --- cast inputs to float64 for accurate special functions / derivatives ---
        xzk64 = xzk.to(torch.float64) # [B, 3]
        x64 = xzk64[..., 0]  # [B]
        z64 = xzk64[..., 1]  # [B]
        logk = xzk64[..., 2]  # [B]
        k64 = torch.pow(10.0, logk)  # [B]
        kabs = k64.abs()  # [B]

        # xs as [B] for broadcasting
        #xs64 = torch.as_tensor(x_s, dtype=torch.float64, device=xzk.device).squeeze(-1)  # [B, 1]
        xs64 = torch.as_tensor(x_s, dtype=torch.float64, device=xzk.device) # [B]

        # get rs vector, from the source to the BC point. zs = 0.0, at the surface
        r_vect = torch.stack([x64 - xs64, z64 - 0.0], dim=-1)  # [B,2]

        # Get distance from the source
        r = torch.linalg.vector_norm(r_vect, ord=2, dim=-1) # [B]
        r = torch.clamp(r, min=1e-12)

        # alpha = k * K1(kr) / K0(kr), with k=0 → alpha=0
        kr = torch.clamp(k64 * r, min=1e-12) # [B]
        K0 = torch.special.modified_bessel_k0(kr)
        K1 = torch.special.modified_bessel_k1(kr)

        # normals
        if bc_type == "bottom":  # n = (0,-1)  => ∂/∂n = -∂/∂z
            n = torch.tensor([0.0, -1.0], dtype=torch.float64, device=xzk.device)  # [2]
        elif bc_type == "left":  # n = (-1,0)  => ∂/∂n = -∂/∂x
            n = torch.tensor([-1.0, 0.0], dtype=torch.float64, device=xzk.device)
        else:  # "right": n = (1,0)  => ∂/∂n = +∂/∂x
            n = torch.tensor([1.0, 0.0], dtype=torch.float64, device=xzk.device)

        # cosθ = (r̂ · n)  with r̂ = r_vect / r
        r_hat = r_vect / r.unsqueeze(-1)  # [B,2]
        cos_theta = (r_hat * n).sum(dim=-1)  # [B]

        # α = k * cosθ * K1(kr)/K0(kr); handle k=0 smoothly
        alpha64 = k64 * cos_theta * K1 / torch.clamp(K0, min=1e-12) # [B]
        alpha = alpha64.unsqueeze(-1)  # [B,1]

        # gradients wrt x,z of u (total secondary potential predicted by PINN)
        grad_u = torch.autograd.grad(u, xzk, grad_outputs=torch.ones_like(u), create_graph=True)[0] # [B,3]
        u_x = grad_u[..., 0:1]  # [B,1]
        u_z = grad_u[..., 1:2]  # [B,1]

        # n·∇(u0+u)
        if bc_type == 'bottom':
            u0, u0_z = u0_components[0], u0_components[1]  # [B,1]
            n_grad = -(u0_z + u_z)              # n·∇ on bottom is -∂/∂z
        else: # bc_type in ('left', 'right'):
            u0, u0_x = u0_components[0], u0_components[1]  # [B,1]
            if bc_type == "left":
                n_grad = -(u0_x + u_x)          # n·∇ on left is -∂/∂x
            else:
                n_grad = (u0_x + u_x)           # n·∇ on right is +∂/∂x

        # n·∇(u0+u) + α (u0+u) = 0
        bc_robin_resid = n_grad + alpha * (u0 + u)  # [B,1]
        return bc_robin_resid.to(torch.float32)
