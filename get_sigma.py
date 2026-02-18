import json
import torch

def _load_sigma_parameters():
    """
    Loads the sigma model parameters from the file 'sigma_model.json' or '../sigma_model.json'
    Return:
        (tuple): (params["sigma0"], params["sigmapeak"], params["std"], params["mean"])
    """
    try:
        with open("sigma_model.json", "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        with open("../sigma_model.json", "r") as f:
            params = json.load(f)
    return params["sigma0"], params["sigmapeak"], params["std"], params["mean"]

def get_sigma_gaussian(xz, from_file=True, grad=True, sigma0=None, sigmapeak=None, std=None, mean=None):
    """
    Return sigma(xz) from the 2D gaussian kernel model. sigma does not change along y axis.
    It gives (sigmapeak - sigma0) distribution! I.e. sigma1(xz) is anomalous part only!
    Reads the sigma gaussian distribution parameters from the file "sigma_model.json":
        sigma0 (float): background conductivity
        sigmapeak (float): peak value conductivity at (mean_x, mean_z)
        std (float): Standard deviation
        mean (list of floats): mean of the Gaussian (mean_x, mean_z)

    Args:
        xz (torch.Tensor): xz coordinates of shape [B, 2]. y does not have impact since it is 2D sigma(xz)
        from_file (bool): if True, it will read the sigma gaussian distribution parameters from "sigma_model.json" file
        grad (bool): if True, calculates the gradient components: (sigma, sigma_x, sigma_y, sigma_z)
        sigma0 (float): background conductivity
        sigmapeak (float): peak value conductivity at (mean_x, mean_z)
        std (float): Standard deviation
        mean (list of floats): mean of the Gaussian (mean_x, mean_z)
    Returns:
        Tuple of tensors of shape [B, 1] (sigma, sigma_x, sigma_z) if grad=True, otherwise returns sigma.
    """
    # Read the sigma gaussian distribution parameters
    if from_file:
        sigma0, sigmapeak, std, mean = _load_sigma_parameters()

    assert xz.dim() == 2 and xz.shape[-1] == 2, f"xz must have shape [B, 2]. current xz.shape: {xz.shape}"
    if grad:
        assert xz.requires_grad, "xz must require grad to calculate gradient components of sigma"

    mean = torch.tensor(mean, dtype=xz.dtype, device=xz.device)
    std = torch.tensor([std, std], dtype=xz.dtype, device=xz.device)
    exponent_term = -0.5 * (((xz - mean) / std) ** 2).sum(dim=-1, keepdim=True)
    gaussian_kernel = torch.exp(exponent_term)

    #sigma = sigma0 + (sigmapeak - sigma0) * gaussian_kernel
    sigma = (sigmapeak - sigma0) * gaussian_kernel

    if grad:
        # Compute gradients
        grad_sigma = torch.autograd.grad(sigma, xz, grad_outputs=torch.ones_like(sigma),
                                         create_graph=True, retain_graph=True)[0]  # Shape [B, 2]
        sigma_x = grad_sigma[..., 0:1]  # Shape [B, 1]
        sigma_z = grad_sigma[..., 1:2]  # Shape [B, 1]
        return sigma, sigma_x, sigma_z
    return sigma

def get_sigma_vertical_fault(xz, from_file=False, grad=True, sigma0=0.01, sigmapeak=0.001, x_fault=0.2, beta=100):
    """
    Return sigma1(xz) (anomalous part) for a Vertical Fault model.
    sigma(x,z) changes smoothly from sigma0 (left) to sigmapeak (right) at x = x_fault.

    Model:
        Medium 1 (x < x_fault): sigma = sigma0
        Medium 2 (x > x_fault): sigma = sigmapeak
        Transition: Sigmoid function controlled by beta.

    Returns sigma1 = sigma_total - sigma0.

    Args:
        xz (torch.Tensor): xz coordinates [B, 2].
        from_file (bool): If True, loads params from json (requires implementation of _load_sigma_parameters or manual passing).
        grad (bool): If True, returns (sigma1, sigma1_x, sigma1_z).
        sigma0 (float): Background conductivity (Medium 1, Left).
        sigmapeak (float): Target conductivity (Medium 2, Right).
        x_fault (float): X-coordinate of the vertical interface.
        beta (float): Sharpness of the transition. Higher = sharper step.
                      (e.g., beta=100 gives a transition width of ~10cm).

    Returns:
        Tuple of tensors [B, 1] (sigma1, sigma1_x, sigma1_z) if grad=True.
    """

    # 1. Handle Parameter Loading
    # If using your existing file loader, you might need to adapt it.
    # Here we assume defaults if not passed explicitly for demonstration.
    if from_file and (sigma0 is None):
        # Placeholder for your loading logic
        # sigma0, sigmapeak, x_fault, beta = _load_vertical_fault_parameters()
        pass

    # Set defaults if None (for safety)
    if x_fault is None: x_fault = 0.5
    if beta is None: beta = 100.0  # Sharpness factor

    assert xz.dim() == 2 and xz.shape[-1] == 2, f"xz must have shape [B, 2]"
    if grad:
        assert xz.requires_grad, "xz must require grad"

    # 2. Extract X coordinate
    x = xz[:, 0:1]  # [B, 1]

    # 3. Calculate Sigmoid Transition
    # We want sigma1 = 0 on Left, sigma1 = (sigmapeak - sigma0) on Right.
    # Sigmoid(v) -> 0 when v is negative (Left of fault)
    # Sigmoid(v) -> 1 when v is positive (Right of fault)

    argument = beta * (x - x_fault)
    transition = torch.sigmoid(argument)

    # 4. Calculate Anomalous Conductivity sigma1
    # sigma_total = sigma0 + (sigmapeak - sigma0) * transition
    # sigma1      = sigma_total - sigma0
    sigma1 = (sigmapeak - sigma0) * transition

    if grad:
        # Compute gradients using Autograd (safest method)
        grad_sigma = torch.autograd.grad(
            sigma1, xz,
            grad_outputs=torch.ones_like(sigma1),
            create_graph=True, retain_graph=True
        )[0]

        sigma1_x = grad_sigma[..., 0:1]
        sigma1_z = grad_sigma[..., 1:2]  # Will be approx 0 for vertical fault

        return sigma1, sigma1_x, sigma1_z
    return sigma1
