from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
#from logger import logger
from scipy.stats import truncexpon, loguniform
from get_sigma import get_sigma_vertical_fault
from physics import get_analytical_solution
from pde_bc import PdeBc
import deepxde as dde
dde.backend.set_default_backend("pytorch")

class Data:
    def __init__(self, device=None):
        """
        Parameters:
        - device (torch.device): The device to store tensors on.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.points = {} # dict[str, torch.Tensor]
        self.dataloaders = {}  # dict[str, DataLoader]
        pass

def add_z_val(tensor_x, z_val):
    return np.hstack((tensor_x, np.full((tensor_x.shape[0], 1), z_val)))

def add_x_val(tensor_z, x_val):
    return np.hstack((np.full((tensor_z.shape[0], 1), x_val), tensor_z))

def add_k(tensor_xz, k_max_val, k_distr, inv_exp_lambda_val=None, seed=None):
    """
    Add wavenumber k to the input tensor
    Args:
        tensor_xz (torch.tensor): tensor of the shape [B, N, 2]
        k_max_val (float): maximum k value
        k_distr (str): ("exponential", "uniform", "loguniform")
        inv_exp_lambda_val (float): scale parameter (1/lambda) of the exponential distribution
        seed (int, optional): random seed for reproducibility
    """
    if k_distr == "exponential":
        k_vals = sample_truncated_exponential(n=tensor_xz.shape[0], scale=inv_exp_lambda_val, max_val=k_max_val, seed=seed)
    elif k_distr == "uniform":
        k_vals = sample_truncated_uniform(n=tensor_xz.shape[0], max_val=k_max_val, seed=seed)
    elif k_distr == "loguniform":
        k_vals = sample_loguniform(n=tensor_xz.shape[0], max_val=k_max_val, seed=seed)
    else:
        raise ValueError(f"Unsupported k_distr: {k_distr}. Use 'exponential', 'uniform', 'loguniform'.")
    return np.hstack((tensor_xz[:, [0]], k_vals, tensor_xz[:, [1]]))

def sample_truncated_uniform(n, max_val, min_val=1e-4, seed=None):
    """
    Draw random samples from a uniform distribution [min_val, max_val].

    Args:
        n (int): number of samples
        max_val (float): upper bound of distribution
        min_val (float): lower bound of distribution
        seed (int, optional): random seed for reproducibility

    Returns:
        np.ndarray: [n, 1] shape, samples in [min_val, max_val]
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(min_val, max_val, size=(n, 1))

def sample_truncated_exponential(n, max_val, min_val=1e-4, scale=0.6, seed=None):
    """
    Draw random samples from a truncated [min_val, max_val] exponential distribution.

    Args:
        n (int): number of samples
        max_val (float): upper bound of distribution
        min_val (float): lower bound of distribution
        scale (float): scale parameter (1/lambda) of exponential
        seed (int, optional): random seed for reproducibility

    Returns:
        np.ndarray: [n, 1] shape, samples in [min_val, max_val]
    """
    assert max_val > min_val > 0, "Require 0 < low < max_val"
    b = (max_val - min_val) / scale
    rng = np.random.default_rng(seed)
    samples = truncexpon(b=b, loc=min_val, scale=scale).rvs(size=n, random_state=rng)
    return samples.reshape(-1, 1)

def sample_loguniform(n, max_val, min_val=1e-4, seed=None):
    """
    Draw random samples from a loguniform distribution [min_val, max_val].

    Args:
        n (int): number of samples
        max_val (float): upper bound of distribution
        min_val (float): lower bound of distribution
        seed (int, optional): random seed for reproducibility

    Returns:
        np.ndarray: [n, 1] shape, samples in [min_val, max_val]
    """
    return loguniform.rvs(a=min_val, b=max_val, size=(n, 1), random_state=seed)

def set_xz_requires_grad(tensor):
    """Put requires_grad only for xz but not on k"""
    x = tensor[:, 0:1].clone().detach().requires_grad_(True)
    z = tensor[:, 2:2 + 1].clone().detach().requires_grad_(True)
    k = tensor[:, 1:2]  # no grad

    new_tensor = torch.cat([x, k, z], dim=1)
    return new_tensor

def select_k_vals(idx_min, idx_max):
    # Load the list of the k values values
    k_lst_full = np.load('k_val_list_full.npy')

    # Select k values
    k_lst = k_lst_full[idx_min:idx_max+1]

    np.save('k_val_list.npy', k_lst.astype(np.float32))
    pass

def generate_data(device, k_idx_range, sigma_0, x_s, n_pde, n_pde_subd, n_bc, distr_pde_file=None, seed=None, sample_method="Hammersley",log=True, logger=None):
    """
    Generate collocation points as 2.5D (x, z, log10k) samples.

    All point sets are returned as tensors of shape [N, 3] on `device`,
    so you can directly do e.g.:

        ds_pde = TensorDataset(data.points_pde_domain)
        loader_pde = DataLoader(ds_pde, batch_size=B, shuffle=True)

    Args:
        device (torch.device): Device to place the tensor on.
        k_idx_range (iterable): (k_idx_min, k_idx_max)
        sigma_0 (float): Background conductivity.
        x_s (float): x position of point source
        n_pde (int): Number of points for the domain.
        n_pde_subd (int): Number of points for subdomain.
        n_bc (int): Number of points for the BC.
        distr_pde_file (str | None): File with 3D histogram model for PDE domain.
        seed (int | None): Random seed for reproducibility.
        sample_method (str): Sampling method (“pseudo”, “LHS”, “Halton”, “Hammersley”, “Sobol”).
        log (bool): If True, log sizes of generated tensors.

    Return:
        Data object with attributes:
          - points_pde_domain   : [N_pde, 3]
          - points_pde_subdomain: [N_subd, 3]
          - points_bc_*         : [N_bc, 3]
          - k_val_lst, logk_val_lst
    """
    # ---- seeding -----------------------------------------------------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(np.random.randint(0, 2**16))

    data = Data(device)

    assert len(k_idx_range) == 2, f'len(k_idx_range) must be 2. Now len(k_idx_range): {len(k_idx_range)}'
    assert k_idx_range[1] >= k_idx_range[0], f'Must have: k_idx_range[1] >= k_idx_range[0]. Now: k_idx_range[1] = {k_idx_range[1]} and  k_idx_range[0]: {k_idx_range[0]}'
    data.k_idx_range = k_idx_range # tuple of k indexes used in the training from k_val_list_full.npy
    data.sigma0 = sigma_0  # Background conductivity, for analytical solution

    domain_lst = (
        "pde_domain",
        "pde_subdomain",
        "bc_top",
        "bc_bottom",
        "bc_left",
        "bc_right",
    )

    # ---- constants / domain -----------------------------------------------
    threshold = 0.005        # distance to exclude points near electrodes
    subdomain_radius = 0.1  # radius of the subdomain

    data.domain = ((-2.0, 2.0), (-2.0, 0.0))
    data.points_s = x_s # x position of point source

    # --- geometries (DeepXDE) ----------------------------------------------
    rect = dde.geometry.Rectangle([-2, -2], [2, 0])
    disc = dde.geometry.Disk((x_s, 0.0), subdomain_radius) & rect
    el_disk = dde.geometry.Disk([x_s, 0.0], threshold) & rect

    pde_domain = rect - disc
    pde_subdomain = disc - el_disk

    # BC geometries
    bc_top_iv = dde.geometry.Interval(-2, 2)
    bc_el_iv = dde.geometry.Interval(x_s - threshold, x_s + threshold)
    bc_top_iv = bc_top_iv - bc_el_iv  # top except electrode vicinity

    bc_bottom_iv = dde.geometry.Interval(-2, 2)  # z = -2
    bc_side_iv = dde.geometry.Interval(-2, 0)    # z in [-2, 0], x fixed later

    # ---- helpers ----------------------------------------------------------
    def _rand_points(geom, n):
        # DeepXDE returns NumPy arrays [N, dim]
        return geom.random_points(n, random=sample_method)

    def _to_torch_xz(arr_np):
        # arr_np: [N, 2] NumPy -> torch [N, 2] float32 on device
        return torch.as_tensor(arr_np, dtype=torch.float32, device=device)

    def add_z_val(x_np, z_val):
        # x_np: [N, 1] np -> [N, 2] with (x, z=z_val)
        z_np = np.full_like(x_np, z_val, dtype=np.float32)
        return np.concatenate([x_np.astype(np.float32), z_np], axis=1)

    def add_x_val(z_np, x_val):
        # z_np: [N, 1] np -> [N, 2] with (x=x_val, z)
        x_np = np.full_like(z_np, x_val, dtype=np.float32)
        return np.concatenate([x_np, z_np.astype(np.float32)], axis=1)

    # ---- sample PDE points (x,z) ------------------------------------------
    pde_domain_np = _rand_points(pde_domain, n_pde)
    pde_subdomain_np = _rand_points(pde_subdomain, n_pde_subd)

    # ---- sample BC points (x or z as 1D) ----------------------------------
    bc_top_x = _rand_points(bc_top_iv, n_bc).reshape(-1, 1)     # -> (x, z=0)
    bc_bottom_x = _rand_points(bc_bottom_iv, n_bc).reshape(-1, 1)  # -> (x, z=-2)
    bc_side_z = _rand_points(bc_side_iv, n_bc).reshape(-1, 1)   # -> (x=±2, z)

    # pack as (x,z) NumPy first
    bc_top_np = add_z_val(bc_top_x, z_val=np.float32(0.0))
    bc_bottom_np = add_z_val(bc_bottom_x, z_val=np.float32(-2.0))
    bc_left_np = add_x_val(bc_side_z, x_val=np.float32(-2.0))
    bc_right_np = add_x_val(bc_side_z, x_val=np.float32(2.0))

    # convert to torch [N,2] float32 on device
    data.points['pde_domain'] = _to_torch_xz(pde_domain_np)
    data.points['pde_subdomain'] = _to_torch_xz(pde_subdomain_np)
    data.points['bc_top'] = _to_torch_xz(bc_top_np)
    data.points['bc_bottom'] = _to_torch_xz(bc_bottom_np)
    data.points['bc_left'] = _to_torch_xz(bc_left_np)
    data.points['bc_right'] = _to_torch_xz(bc_right_np)

    # ---- load full k-list, build log10k ----------------------------------------
    with open("k_val_list_full.npy", "rb") as f:
        k_val_lst_np_full = np.load(f)
    data.k_val_lst_full = torch.as_tensor(k_val_lst_np_full, dtype=torch.float32, device=device)
    data.logk_val_lst_full = torch.log10(data.k_val_lst_full)

    # ---- get k-list of selected values only ----------------------------------------
    data.k_val_lst = data.k_val_lst_full[data.k_idx_range[0]:data.k_idx_range[1]+1]
    data.logk_val_lst = torch.log10(data.k_val_lst)

    # uniform probability over k-values
    pK = torch.ones_like(data.logk_val_lst, device=device)

    def _add_logk(xz_tensor: torch.Tensor) -> torch.Tensor:
        """Given [N,2] (x,z) -> [N,3] (x,z,log10k_sampled)."""
        N = xz_tensor.shape[0]
        idx = torch.multinomial(pK, N, replacement=True)
        logk_tensor = data.logk_val_lst[idx].unsqueeze(1)  # [N,1]
        return torch.cat([xz_tensor, logk_tensor], dim=1)

    # ---- optionally override PDE domain from histogram model --------------
    if distr_pde_file is not None:
        # sampler is assumed to return [N,3] = (x,z,log10k)
        sampler = Hist3DResampler.load(distr_pde_file, map_device=device)
        xzk = sampler.sample(n_new=n_pde, seed=seed)  # [N,3] on device

        # filter out points inside the subdomain radius around source in (x,z)
        x = xzk[..., 0]
        z = xzk[..., 1]
        mask = torch.sqrt((x - x_s) ** 2 + z ** 2) > subdomain_radius
        data.points['pde_domain'] = xzk[mask]  # already [N,3]
    else:
        # standard case: PDE domain also starts as [N,2] and needs log10k added
        data.points['pde_domain'] = _add_logk(data.points['pde_domain'])

    # subdomain + BCs always get logk added
    for domain in domain_lst[1:]:
        data.points[domain] = _add_logk(data.points[domain])

    # ---- final safety check (no points too close to source) ---------------
    ref = torch.tensor([x_s, 0.0], dtype=torch.float32, device=device)
    threshold_sq = threshold ** 2

    for domain in domain_lst:
        arr = data.points[domain] # [B,3] (x,z,logk)
        diff = arr[..., :2] - ref  # distance in (x,z) only
        d2 = torch.sum(diff * diff, dim=-1)
        assert torch.all(d2 > threshold_sq), f"{domain} has points near the source."
        assert arr.dtype == torch.float32

    # ---- logging -----------------------------------------------------------
    if log:
        logger.info(
            "Data generated (method=%s):\n%s",
            sample_method if distr_pde_file is None else f"Distribution from file {distr_pde_file}", # sample_method
            "\n".join(
                f"  - {name}: {data.points[name].shape[0]}"
                for name in domain_lst
            )
        )

    return data

def add_new_k_by_copy(tensor_orig, k_old, k_new, logger):
    """
    Create new (x, z, k_new) points by copying all (x, z) locations corresponding to a given existing k_old in a tensor of shape [B, 3].

    Args:
        tensor_orig (torch.Tensor): Input tensor of shape [B, 3] = (x, z, log10k).
        k_old (float): The existing k value whose (x, z) rows will be duplicated.
        k_new (float): The new k value assigned to the duplicated (x, z) rows.

    Returns:
        torch.Tensor:
            A new tensor containing the original rows and the newly created (x, z, log10k_new) rows.
            If no rows match k_old, the original tensor is returned unchanged.
    """
    # 1. Mask for the old k
    mask = tensor_orig[:, 2].isclose(torch.tensor(k_old, dtype=tensor_orig.dtype, device=tensor_orig.device))
    if not mask.any():
        logger.warning(f"No points found for this log10k value: {k_old:.2f}. New k value was not added!!!")
        return tensor_orig

    # 2. Extract xz of those rows
    xz = tensor_orig[mask, :2]  # [N, 2]

    # 3. Create new rows (xz, k_new)
    k_col = torch.full((xz.shape[0], 1), k_new, dtype=tensor_orig.dtype, device=tensor_orig.device)
    new_rows = torch.cat([xz, k_col], dim=1)  # [N, 3]

    # 4. Append
    tensor_new = torch.cat([tensor_orig, new_rows], dim=0)
    return tensor_new

def expand_k_val(data, k_old, k_new, logger):
    """
    Add new (x, z, k_new) data.points by copying all (x, z) locations corresponding to a given existing k_old in a tensor of shape [B, 3].

    Args:
        data: Object with attribute `points`, a dict-like mapping from domain names to torch.Tensors of shape [B, 3].
        k_old (float): Existing k value to search for in each domain tensor.
        k_new (float): New k value that will be created by duplicating the (x, z) positions corresponding to k_old.

    Returns:
            All tensors inside data.points are updated in-place.
    """
    for attr in data.points.keys():
        data.points[attr] = add_new_k_by_copy(data.points[attr], k_old, k_new, logger)
    pass

def make_dataloaders(data, batch_sizes_dict, shuffle=True, drop_last=True, pin_memory=True):
    """
    Makes downloaders
    Args:
        batch_sizes_dict: dict mapping domain name -> batch size, e.g.
            {
                "pde_domain": 2048,
                "pde_subdomain": 2048,
                "bc_top": 512,
                ...
            }
    Return:
        data.dataloaders: (dict of torch.utils.data.dataloader.DataLoader)
    """
    data.dataloaders = {}  # reset

    for key, xzk in data.points.items():
        bs = batch_sizes_dict.get(key, None)
        if bs is None:
            # skip domains where you don't want a loader
            continue
        assert bs <= xzk.shape[0], f"Domain {key}: batch size: {bs} cannot be greater than the tensor size: {xzk.shape[0]}"

        ds = TensorDataset(xzk)  # xzk: [N, 3]
        dl = DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            generator=torch.Generator(device=data.device)
        )
        data.dataloaders[key] = dl

class Hist3DResampler:
    """
    # ---------- Usage ----------

    # (1) Fit & save
    # xzk: [N,3] on CPU or GPU; k_val_lst: sorted 1D tensor of allowed k's
    model_sampling = Hist3DResampler(bins_per_axis=16, k_val_lst=data.logk_val_lst)
    model_sampling.fit(data_orig_rmas.points_pde_domain)
    model_sampling.save("xzk_hist_model.pt")

    # (2) Load & sample later
    model_sampling = Hist3DResampler.load("xzk_hist_model.pt", map_device=data.device)
    xzk_new = model_sampling.sample(n_new=10000, seed=123)

    # (3) Wrap as a dataset / dataloader
    def make_dataset(samples: torch.Tensor) -> TensorDataset:
        # Return all 3 coords; adapt to your pipeline if you split inputs/targets.
        return TensorDataset(samples)

    ds = make_dataset(samples)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    """
    def __init__(self, bins_per_axis=16, k_val_lst=None):
        self.bins = bins_per_axis
        self.k_val_lst = None if k_val_lst is None else k_val_lst.clone()

        # filled by fit() / load()
        self.edges = None          # list of 3 tensors (edges per axis)
        self.probs = None          # flattened [bins^3]
        self.dtype = None
        self.device = None

    @staticmethod
    def _digitize(col, e):
        idx = torch.bucketize(col, e, right=False) - 1
        return idx.clamp(0, len(e) - 2)

    def fit(self, xzk: torch.Tensor):
        """Build a 3D histogram from data xzk: [N,3]."""
        assert xzk.ndim == 2 and xzk.size(1) == 3
        self.dtype, self.device = xzk.dtype, xzk.device

        mins, _ = xzk.min(0)
        maxs, _ = xzk.max(0)
        self.edges = [
            torch.linspace(mins[i], maxs[i], self.bins + 1, dtype=self.dtype, device=self.device)
            for i in range(3)
        ]

        b0 = self._digitize(xzk[:, 0], self.edges[0])
        b1 = self._digitize(xzk[:, 1], self.edges[1])
        b2 = self._digitize(xzk[:, 2], self.edges[2])

        flat_idx = b0 * (self.bins**2) + b1 * self.bins + b2
        K = self.bins ** 3
        counts = torch.bincount(flat_idx, minlength=K).float()
        self.probs = counts / counts.sum()

        if self.k_val_lst is not None:
            self.k_val_lst = self.k_val_lst.to(dtype=self.dtype, device=self.device)

    def sample(self, n_new=1000, seed: int | None = None):
        """Draw n_new samples; snap k to nearest value in k_val_lst if provided."""
        assert self.edges is not None and self.probs is not None, "Call fit() or load() first."
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        else:
            g = None

        chosen = torch.multinomial(self.probs, n_new, replacement=True, generator=g)

        i0 = chosen // (self.bins**2)
        r = chosen % (self.bins**2)
        i1 = r // self.bins
        i2 = r % self.bins

        u = torch.rand(n_new, 3, dtype=self.dtype, device=self.device, generator=g)
        out = torch.empty(n_new, 3, dtype=self.dtype, device=self.device)

        for j, (ij, ej) in enumerate(zip([i0, i1, i2], self.edges)):
            lo = ej[ij]
            hi = ej[ij + 1]
            out[:, j] = lo + (hi - lo) * u[:, j]

        # Snap k (3rd coord) to nearest in k_val_lst
        if self.k_val_lst is not None:
            k_vals = self.k_val_lst.view(1, -1)             # [1, Kk]
            k = out[:, 2:3]                                  # [n, 1]
            nearest_idx = torch.argmin(torch.abs(k - k_vals), dim=1)
            out[:, 2] = self.k_val_lst[nearest_idx]

        return out

    def save(self, path: str):
        """Save histogram ‘model’."""
        state = {
            "bins": self.bins,
            "edges": [e.detach().cpu() for e in self.edges],
            "probs": self.probs.detach().cpu(),
            "k_val_lst": None if self.k_val_lst is None else self.k_val_lst.detach().cpu(),
            "dtype": str(self.dtype),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, map_device: str | torch.device = "cpu"):
        """Load histogram model and map to device."""
        s = torch.load(path, map_location=map_device)
        obj = cls(bins_per_axis=s["bins"], k_val_lst=s["k_val_lst"])
        obj.edges = [e.to(map_device) for e in s["edges"]]
        obj.probs = s["probs"].to(map_device)
        obj.dtype = obj.probs.dtype
        obj.device = torch.device(map_device)
        if obj.k_val_lst is not None:
            obj.k_val_lst = obj.k_val_lst.to(map_device)
        return obj

def reset_iterators(data):
    data.iters = {name: iter(dl) for name, dl in data.dataloaders.items()}
    pass

def prepare_batch(xzk_batch, device):
    """
    Args:
        xzk_batch (torch.tensor): [B, 3] = (x, z, log10k)
        device (torch.device): Device to place the tensor on.
    Returns:
        xzk_for_model: [B, 3] with (x,z) requiring grad, logk no grad
        TODO : there is option to return xz: [B, 2] view used for autograd.grad (du/dx, du/dz)
    """
    # move to device
    xzk_batch = xzk_batch.to(device, non_blocking=True)

    # split
    xz = xzk_batch[:, :2].clone().detach().requires_grad_(True)  # grad on
    logk = xzk_batch[:, 2:3].clone().detach()                    # grad off

    # merge back for model convenience
    xzk_for_model = torch.cat([xz, logk], dim=-1)  # [B,3]
    return xzk_for_model
    #return xzk_for_model, xz

def get_prepared_batch(data, domain_name):
    """
    Returns prepared batch xzk_for_model for given domain.
    Restarts iterator when it is exhausted.
    """
    dl = data.dataloaders[domain_name]
    it = data.iters[domain_name]

    try:
        (xzk_raw,) = next(it)          # xzk_raw: [B, 3]
    except StopIteration:
        # restart this domain iterator
        data.iters[domain_name] = iter(dl)
        (xzk_raw,) = next(data.iters[domain_name])
    return prepare_batch(xzk_raw, data.device)

def get_prepared_batches(data):
    """
    Get the prepared batches in the dict data.batch
    """
    data.batch = {}
    for name in data.points.keys():
        data.batch[name] = get_prepared_batch(data, name)  # [B, 3]
    pass

def save_train_tensors(data, folder_name):
    path = Path(folder_name)
    path.mkdir(parents=True, exist_ok=True)
    for domain in data.points.keys():
        tensor = data.points[domain]
        torch.save(tensor, path / f"{domain}.pt")

def load_train_tensors(data, logger=None, tensor_dir=None, log=True):
    if tensor_dir is None:
        tensor_dir = Path("train_tensors")
    if not tensor_dir.exists():
        logger.info(f"No saved tensor data found at '{tensor_dir}'. Skipping tensor loading.")
        return

    # Load each tensor
    for domain in data.points.keys():
        loaded = torch.load(tensor_dir / f"{domain}.pt", map_location=data.device)
        data.points[domain] = loaded

    # Summarize counts
    pde_attrs = [a for a in data.points.keys() if a.startswith('pde')]
    bc_attrs  = [a for a in data.points.keys() if a.startswith('bc')]

    total_pde = sum(data.points[a].shape[0] for a in pde_attrs)
    total_bc  = sum(data.points[a].shape[0] for a in bc_attrs)

    if log:
        logger.info(f"Loaded {total_pde} PDE point and {total_bc} BC points from {tensor_dir}")
        for a in pde_attrs + bc_attrs:
            count = data.points[a].shape[0]
            label = a.replace('points_', '').replace('_', ' ').title()
            logger.info(f"  {label}: {count}")

def add_bad_points(data_test, model, data_train, num, attr, tolerance, logger):
    """
    Add up to the top num of points (by absolute residuals) from test_tensor into train_datapoints, but only those with abs(residual) > tolerance.
    data_test: Data object providing .points[attr] (shape [B, 3])
    model: pytorch network model with forward(xzk) with k=log10k
    data_train: Data object owning .points[attr] (shape [M, 3])
    num (int): max number of new points to add
    attr (str): one of ['pde_domain', 'pde_subdomain', 'bc_top']
    tolerance (float): minimal absolute residual to consider
    """
    assert attr in ("pde_domain", "pde_subdomain", "bc_top"), f"Wrong attribute: {attr}"
    pdebc = PdeBc()

    # 1) get collocation points [B, 3]
    # xzk = getattr(data_test, f"batch_{attr}")          # [B, 3]
    xzk = data_test.batch[attr] # [B, 3]
    B, D = xzk.shape
    assert D == 3, f"Expected (x,z, log10k) with D=3, got D={D}"

    # 2) model predictions and residuals
    u1 = model.forward(xzk)  # [B, 1] expected
    if attr.startswith("pde"):
        u0_components = get_analytical_solution(x_s=data_test.points_s, xzk=xzk, sigma0=data_test.sigma0)
        resid = pdebc.pde(u=u1, u0_components=u0_components, sigma_components=get_sigma_vertical_fault(xz=xzk[..., 0:2]),
            xzk=xzk, x_s=data_train.points_s, sigma0=data_test.sigma0)  # [B, 1]
    else:  # bc_top
        resid = pdebc.bc_no_flow(u=u1, sigma=get_sigma_vertical_fault(xz=xzk[..., 0:2], grad=False),
            sigma0=data_test.sigma0, xzk=xzk)  # [B, 1]

    # 3) flatten residuals and indices
    resid = resid.detach()
    xzk = xzk.detach()

    abs_vals: torch.Tensor = resid.abs().reshape(-1)  # [B, 1]
    flat_xzk = xzk                      # [B, 3]

    # 4) keep only above tolerance
    mask_idx = (abs_vals > tolerance).nonzero(as_tuple=False).flatten()  # [K]

    if mask_idx.numel() == 0:
        logger.info(f"[{attr}] nothing above tolerance={tolerance:g}")
        return 0

    # 5) select top-k by |residual|
    k_sel = min(num, mask_idx.numel())

    top_in_mask = abs_vals[mask_idx].topk(k_sel, dim=0, largest=True).indices
    top_flat_idx = mask_idx[top_in_mask]        # [k_sel]
    coords = flat_xzk.index_select(0, top_flat_idx)                # [k_sel, 2]

    # 6) append into data_train.points[attr]
    target_name = attr
    train_pts = data_train.points[target_name]
    #train_pts = getattr(data_train, target_name)                  # [M, 2]
    coords = coords.to(device=train_pts.device, dtype=train_pts.dtype)
    #setattr(data_train, target_name, torch.cat([train_pts, coords], dim=0))
    data_train.points[target_name] = torch.cat([train_pts, coords], dim=0)
    added_count = coords.size(0)

    # 7) logging summary (residuals, x/z range, k range, and log10(k) range if valid)
    sel_res = abs_vals[top_flat_idx]
    # k range
    logk_min = coords[..., 2].min()
    logk_max = coords[..., 2].max()

    logger.info(
        f"[{attr}] added {added_count} points; "
        f"resid: [{sel_res.min().item():.2e}, {sel_res.max().item():.2e}] "
        f"k: [{logk_min:.2e}, {logk_max:.2e}]"
        # f"x: [{coords[:,0].min().item():.3g}, {coords[:,0].max().item():.3g}], "
        # f"z: [{coords[:,1].min().item():.3g}, {coords[:,1].max().item():.3g}], "
    )
    return added_count

def run_rmas(model, data, num_lst, tolerance, logger):
    """
    Currently it is just residual-based adaptive refinement (RAR) algorithm
    TODO: RMAS (Residual multipeaks adaptive sampling) over the PDE domains
    Args:
        model: pytorch network model with forward(xzk) with k=log10k
        data: Data object owning .points[attr] (shape [M, 3])
        num_lst (iterable): number of points per domain with the top highest PDE residual for each domain
                            ['pde_domain', 'pde_subdomain', 'bc_top']
        tolerance (float): minimal absolute residual to consider
    Return:
        count_bad_point_lst: list of number of added bad points
        Update tensors in data: data.points_***
    """
    # Generate and prepare a fresh test set
    assert len(num_lst) == 3
    #num_lst.extend([num_lst[-1]] * 3) # Extend the list for the robin BC
    data_test = generate_data(data.device, k_idx_range=data.k_idx_range, sigma_0=data.sigma0, x_s=data.points_s, n_pde=50000, n_pde_subd=10000, n_bc=5000, log=False)
    data_test.batch_sizes_dict = {domain: pts.shape[0] for domain, pts in data_test.points.items()}  # Full dataset
    make_dataloaders(data_test, data_test.batch_sizes_dict, shuffle=False, pin_memory=not data_test.points['pde_domain'].is_cuda)
    reset_iterators(data_test)
    get_prepared_batches(data_test)

    # For each PDE domain run add_bad_point
    count_bad_point_lst = []
    attr_lst = ['pde_domain', 'pde_subdomain', 'bc_top']
    for attr, num in zip(attr_lst, num_lst):
        # add the worst point (if any) and store
        count_bad_points = add_bad_points(data_test=data_test, model=model, data_train=data, num=num, attr=attr, tolerance=tolerance, logger=logger)
        count_bad_point_lst.append(count_bad_points)
    return count_bad_point_lst

def build_cache_tensors(data):
    """Precompute and store primary potentials, their derivatives, and sigma(+grads)
       for PDE and BC collocation points into data.*_cache dictionaries."""
    with torch.no_grad():
        u0, u0_x, u0_z, u0_xx, u0_zz = get_analytical_solution(data.points_s, data.points_pde_domain, sigma0=data.sigma0)
    sigma, sigma_x, sigma_z = get_sigma_vertical_fault(data.points_pde_domain)
    data.points_pde_domain_cache = {'u0':u0, 'u0_x':u0_x, 'u0_z':u0_z, 'u0_xx':u0_xx, 'u0_zz':u0_zz,
                                    'sigma':sigma.detach(), 'sigma_x':sigma_x.detach(), 'sigma_z':sigma_z.detach()}
    with torch.no_grad():
        u0, u0_x, u0_z, u0_xx, u0_zz = get_analytical_solution(data.points_s, data.points_pde_subdomain, sigma0=data.sigma0)
    sigma, sigma_x, sigma_z = get_sigma_vertical_fault(data.points_pde_subdomain)
    data.points_pde_subdomain_cache = {'u0':u0, 'u0_x':u0_x, 'u0_z':u0_z, 'u0_xx':u0_xx, 'u0_zz':u0_zz,
                                    'sigma':sigma.detach(), 'sigma_x':sigma_x.detach(), 'sigma_z':sigma_z.detach()}

    sigma = get_sigma_vertical_fault(data.points_bc_top, grad=False)
    data.points_bc_top_cache = {'sigma':sigma.detach()}

    # for Robin BC, bottom and sides
    with torch.no_grad():
        u0, _, u0_z, _, _ = get_analytical_solution(x_s=data.points_s, xzk=data.points['bc_bottom'], sigma0=data.sigma0, out_dtype=torch.float64)
    data.points_bc_bottom_cache = {'u0':u0, 'u0_z':u0_z}

    with torch.no_grad():
        u0, u0_x, _, _, _ = get_analytical_solution(x_s=data.points_s, xzk=data.points['bc_left'], sigma0=data.sigma0, out_dtype=torch.float64)
    data.points_bc_left_cache = {'u0': u0, 'u0_x': u0_x}

    with torch.no_grad():
        u0, u0_x, _, _, _ = get_analytical_solution(x_s=data.points_s, xzk=data.points['bc_right'], sigma0=data.sigma0, out_dtype=torch.float64)
    data.points_bc_right_cache = {'u0': u0, 'u0_x': u0_x}
    pass
