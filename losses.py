import torch
from pde_bc import PdeBc
from get_sigma import get_sigma_vertical_fault
from physics import get_analytical_solution

class Losses:
    def __init__(self, loss_weights):
        """
        Args:
            loss_weights (torch.Tensor): tensor of loss weights
        """
        self.loss_weights = loss_weights
        self.assert_loss_weights()
        self.pde_bc = PdeBc()
        pass

    def assert_loss_weights(self):
        assert self.loss_weights.shape[0] == 4

    def get_all_losses(self, model, data):
        """
        Computes total loss, sum of MSEs
        Args:
            model: pytorch network model of u(x, z, log10k)
            data: Data object with data
        Return:
            (total_loss, all_losses) = log(loss_weights * loss_lst), all losses (unweighted)
            total_loss | "loss_pde_domain", "loss_pde_subdomain", "loss_bc_top", "bc_robin_loss"
        """
        x_s = data.points_s

        # PDE loss
        xzk = data.batch['pde_domain']
        u0_domain = get_analytical_solution(x_s=x_s, xzk=xzk, sigma0=data.sigma0, get_residual=False)
        sigma_domain = get_sigma_vertical_fault(xz=xzk[..., 0:2])
        pde_domain_loss = self.pde_loss(model, x_s=x_s, xzk=xzk, sigma0=data.sigma0, u0_components=u0_domain, sigma_components=sigma_domain)

        xzk = data.batch['pde_subdomain']
        u0_subdomain = get_analytical_solution(x_s=x_s, xzk=xzk, sigma0=data.sigma0, get_residual=False)
        sigma_subdomain = get_sigma_vertical_fault(xz=xzk[..., 0:2])
        pde_subdomain_loss = self.pde_loss(model, x_s=x_s, xzk=xzk, sigma0=data.sigma0, u0_components=u0_subdomain, sigma_components=sigma_subdomain)

        # BC loss
        xzk = data.batch['bc_top']
        sigma = get_sigma_vertical_fault(xz=xzk[..., 0:2], grad=False)
        bc_top_loss = self.bc_no_flow_loss(model, x_s=x_s, xzk=xzk, sigma=sigma, sigma0=data.sigma0)

        xzk = data.batch['bc_bottom']
        u0, _, u0_z, _, _ = get_analytical_solution(x_s=x_s, xzk=xzk, sigma0=data.sigma0, get_residual=False)
        u0_bc_bottom = (u0, u0_z)
        bc_bottom_loss = self.bc_robin_loss(model, x_s=x_s, xzk=xzk, u0_components=u0_bc_bottom, bc_type='bottom')

        xzk = data.batch['bc_left']
        u0, u0_x, _, _, _ = get_analytical_solution(x_s=x_s, xzk=xzk, sigma0=data.sigma0, get_residual=False)
        u0_bc_left = (u0, u0_x)
        bc_left_loss = self.bc_robin_loss(model, x_s=x_s, xzk=xzk, u0_components=u0_bc_left, bc_type='left')

        xzk = data.batch['bc_right']
        u0, u0_x, _, _, _ = get_analytical_solution(x_s=x_s, xzk=xzk, sigma0=data.sigma0, get_residual=False)
        u0_bc_right = (u0, u0_x)
        bc_right_loss = self.bc_robin_loss(model, x_s=x_s, xzk=xzk, u0_components=u0_bc_right, bc_type='right')

        # Total loss
        bc_robin_loss = data.sigma0 * (bc_bottom_loss + bc_left_loss + bc_right_loss) / 3
        all_losses = torch.stack([pde_domain_loss, pde_subdomain_loss, bc_top_loss, bc_robin_loss])

        ## Trainable weighted loss terms
        eps = 0.01
        denom = eps ** 2 + self.loss_weights ** 2
        weighted = torch.dot(0.5 / denom, all_losses)
        reg = torch.log(denom)
        total_loss = torch.sum(weighted + reg)
        ################

        all_losses = all_losses.detach()
        return total_loss, all_losses

    def pde_loss(self, model, x_s, xzk, sigma0, u0_components, sigma_components):
        """
        sum((PDE_residual)^2)/num_of_points
        Calculates PDE residual for the Poisson equation with current point source. The singularity removal is done with decomposing potential into primary and secondary.
        ∇(σ₀ ∇φ₁) + ∇(σ₁ ∇φ₀) + ∇(σ₁ ∇φ₁) - k² (σ₀ φ₁ + σ₁ φ₀ + σ₁ φ₁) = 0
        Args:
            model: pytorch network model of φ₁(x, z, log10k)
            x_s (float): position of the electrode, Source x-location (z_s = 0).
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
            sigma0 (float): background σ₀
            u0_components (iterable): (u0, u0_x, u0_z, u0_xx, u0_zz) φ₀ of shapes [B, 1], the analytically computed primary potential and its derivatives
            sigma_components (iterable): (sigma, sigma_x, sigma_z) of shapes [B, 1], the σ₁, heterogeneity
        Returns:
            (torch.Tensor): PDE loss (tensor of one element).
        """
        # Secondary potential
        u = model.forward(xzk)

        # Get PDE residual
        pde_residual = self.pde_bc.pde(u, u0_components=u0_components, sigma_components=sigma_components, xzk=xzk, x_s=x_s, sigma0=sigma0)

        # Get MSE Loss
        pde_loss = pde_residual.square().mean()
        #pde_loss = pde_residual.to(torch.float64).square().mean().to(torch.float32) # It might be better
        return pde_loss.to(dtype=torch.float32)

    def bc_no_flow_loss(self, model, x_s, xzk, sigma, sigma0):
        """
        sum(bc_no_flow^2)/num_of_points
        Args:
            model: pytorch network model of φ₁(x, z, log10k)
            x_s (float): position of the electrode, Source x-location (z_s = 0).
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
            sigma (torch.Tensor): of shape [B, 1], sigma(x, z).
            sigma0 (float): background electrical conductivity
        Returns:
            (torch.Tensor): BC no-flow loss (tensor of one element).
        """
        # Calculate the secondary potential u
        u_bc = model.forward(xzk)

        # Get BC no-flow residual
        bc_no_flow_resid = self.pde_bc.bc_no_flow(u=u_bc, sigma=sigma, sigma0=sigma0, xzk=xzk)

        # Get MSE loss
        bc_no_flow_loss = bc_no_flow_resid.square().mean()
        return bc_no_flow_loss.to(dtype=torch.float32)

    def bc_robin_loss(self, model, x_s, xzk, u0_components, bc_type):
        r"""
        sum((BC_Robin_residual) ^ 2) / num_of_points
        Args:
            model: pytorch network model of φ₁(x, z, log10k)
            x_s (float): position of the electrode, Source x-location (z_s = 0).
            xzk (torch.Tensor): xz,log10k tensor with xz coordinates and log10k wavenumbers of shape [B, 3]
            u0_components (iterable): (u0, u0_x) or (u0, u0_z) of shapes [B, 1], the analytically computed primary potential and its derivatives. It is better be in float64
            bc_type (str): 'bottom', 'left', or 'right'
        Returns:
            (torch.Tensor): Robin BC loss (tensor of one element).
        """
        # Secondary potential
        u = model.forward(xzk)

        # Get BC Robin residual
        bc_robin_resid = self.pde_bc.bc_robin(u, u0_components=u0_components, xzk=xzk, x_s=x_s, bc_type=bc_type)

        # Get MSE loss
        bc_robin_loss = bc_robin_resid.square().mean()
        return bc_robin_loss.to(dtype=torch.float32)
