import pathlib
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8-colorblind')
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Arc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from evaluation import create_xyz
from pde_bc import PdeBc
from get_sigma import get_sigma_vertical_fault
from physics import get_analytical_solution

import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def plot_data_points(self, data_obj):
        """
        Args:
            data_obj: Data object contains tensors data.points
        """
        fig = go.Figure()

        for attr in data_obj.points.keys():
            pts = data_obj.points[attr]

            x = pts[:, 0].cpu().numpy()
            z = pts[:, 1].cpu().numpy()
            logk = pts[:, 2].cpu().numpy()

            fig.add_trace(go.Scatter3d(
                x=x,
                y=logk,
                z=z,
                mode="markers",
                marker=dict(size=1),
                name=attr,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"  # hover title
                    "x: %{x:.3f}<br>"
                    "log10k: %{y:.2f}<br>"
                    "z: %{z:.3f}<br>"
                    "<extra></extra>"  # remove extra box
                )
            ))

        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="log10k", zaxis_title="z"),
            width=600, height=600
        )

        fig.show()
        return fig

    def plot_losshistory(self, losshistory, sample_rate=1, start_idx=0, ax=None):
        """
        Plot loss history
        Args:
            losshistory (losshistory object)
            sample_rate (int): sample rate to avoid clutter (default 1 means all points)
            start_idx (int): index of the epoch to start with
            ax (matplotlib.axes._axes.Axes)
        """
        loss_name_lst = tuple(losshistory.losshistory.keys())[2:]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))

        # Plot total_loss on primary axis
        ax.plot(losshistory.losshistory['epoch'][start_idx:][::sample_rate],
                losshistory.losshistory['total_loss'][start_idx:][::sample_rate],
                label='total_loss', linewidth=2, color='red')

        # Create secondary axis for the other losses
        ax2 = ax.twinx()
        for loss_name in loss_name_lst:
            ax2.plot(losshistory.losshistory['epoch'][start_idx:][::sample_rate],
                     np.log(np.maximum(losshistory.losshistory[loss_name][start_idx:][::sample_rate]), 1e-20),
                     label=loss_name, linestyle='--')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax2.set_ylabel(r'$Log_e(\text{Loss})$')
        ax.set_title('Losses')

        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)

    def plot_resipy_pinn(self, df_res, ax=None):
        """Plot Resipy vs PINN Rhoa"""
        rms = np.sqrt(np.mean(np.square(df_res['Abs_diff'])))

        vmin = df_res[['Rhoa', 'Rhoa_pinn']].min().min()
        vmax = df_res[['Rhoa', 'Rhoa_pinn']].max().max()

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(df_res['Rhoa'], df_res['Rhoa_pinn'], 'o', markersize=4, alpha=0.7)

        formatter = FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(formatter)

        ax.set_aspect('equal')
        ax.set_xlim((vmin, vmax))
        ax.set_ylim((vmin, vmax))
        ax.set_xlabel(r'$\text{Resipy }\rho_a, \Omega\cdot m$')
        ax.set_ylabel(r'$\text{PINN }\rho_a, \Omega\cdot m$')
        ax.set_title(f'PINN vs Resipy.\n rms={rms:.2f} $\\Omega \\cdot m$', fontsize=9)

    def plot_losshistory_from_logfile(self, losshistory, together, ax=None):
        """
        Plot the losses all, in one plot
        Args:
            losshistory (np.ndarray): [N, 11] [epoch, train_losses, test_losses]
            together (boolean): if True, all in one plot
            ax (matplotlib.axes._axes.Axes)
        """
        # 0, 1-7, 8-12
        loss_name_lst = ("total_loss", "loss_pde_backgr", "loss_pde_foregr", "loss_pde_subd1", "loss_pde_subd2", "loss_bc_backgr", "loss_bc_foregr")

        if together:
            colors = ["black", "blue", "red", "green", "gray", "yellow", "lime"]
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 2))
            for idx, loss_name in enumerate(loss_name_lst, start=1):
                color = colors[idx - 1]
                ax.plot(losshistory[:, 0], losshistory[:, idx], color=color, linestyle='-', linewidth=1,
                        label=f'Train {loss_name}')
                ax.plot(losshistory[:, 0], losshistory[:, idx + 5], color=color, linestyle='--', linewidth=1,
                        label=f'Test {loss_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(r'$Log_e(Loss)$')
            ax.legend(bbox_to_anchor=(1.01, 1.10), loc='upper left', fontsize=8)
            ax.set_title('Losses')
            return
        else:
            if ax is None:
                fig, ax = plt.subplots(nrows=7, figsize=(12, 7 * 1.5))
            for idx, axis in enumerate(ax):
                axis.plot(losshistory[:, 0], losshistory[:, idx + 1], linestyle='-', linewidth=1,
                        label=f'Train {loss_name_lst[idx]}')
                axis.plot(losshistory[:, 0], losshistory[:, idx + 1 + 7], linestyle='--', linewidth=1,
                        label=f'Test {loss_name_lst[idx]}')
                axis.set_xlabel('Epoch')
                axis.set_ylabel(r'$Log_e(Loss)$')
                #axis.legend(loc='upper right', fontsize=8)
                axis.legend(bbox_to_anchor=(1.01, 1.10), loc='upper left', fontsize=8)
                axis.set_title(f'{loss_name_lst[idx]}')
            plt.tight_layout()
            return

    def plot_losshistory_from_logfile_plotly(self, losshistory, together):
        """
        Plot losses using Plotly.

        Args:
            losshistory (np.ndarray): Assumed to be of shape [N, 13] where:
                - Column 0: Epoch.
                - Columns 1-7: Training losses.
                - Columns 8-14: Test losses.
            together (bool): If True, all losses are plotted on one figure.
        """
        loss_name_lst = ("total_loss", "loss_pde", "loss_pde_subd", "loss_bc_top", "loss_bc_robin")
        colors = ["black", "blue", "red", "green", "gray", "yellow", "lime"]

        if together:
            fig = go.Figure()
            # Loop over the losses: for each loss add training (solid) and test (dashed) traces.
            for idx, loss_name in enumerate(loss_name_lst, start=1):
                fig.add_trace(go.Scatter(
                    x=losshistory[:, 0],
                    y=losshistory[:, idx],
                    mode='lines',
                    line=dict(color=colors[idx - 1], width=1),
                    name=f'Train {loss_name}',
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"  # hover title
                        "epoch: %{x:.0f}<br>"
                        "loss: %{y:.2e}<br>"
                        "<extra></extra>"  # remove extra box
                    )
                ))
                fig.add_trace(go.Scatter(
                    x=losshistory[:, 0],
                    y=losshistory[:, idx + 5],
                    mode='lines',
                    line=dict(color=colors[idx - 1], width=1, dash='dash'),
                    name=f'Test {loss_name}',
                    hovertemplate = (
                        "<b>%{fullData.name}</b><br>"  # hover title
                        "epoch: %{x:.0f}<br>"
                        "loss: %{y:.2e}<br>"
                        "<extra></extra>"  # remove extra box
                    )
                ))
            fig.update_layout(
                title='Losses',
                xaxis_title='Epoch',
                yaxis_title='Log10(Loss)',
                legend=dict(font=dict(size=8))
            )
        else:
            # Create 5 vertical subplots, one per loss
            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True,
                subplot_titles=loss_name_lst
            )
            for i, loss_name in enumerate(loss_name_lst):
                row = i + 1
                fig.add_trace(go.Scatter(
                    x=losshistory[:, 0],
                    y=losshistory[:, i + 1],
                    mode='lines',
                    line=dict(width=1, color='blue'),
                    name=f'Train {loss_name}',
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"  # hover title
                        "epoch: %{x:.0f}<br>"
                        "loss: %{y:.2e}<br>"
                        "<extra></extra>"  # remove extra box
                    )
                ), row=row, col=1)
                fig.add_trace(go.Scatter(
                    x=losshistory[:, 0],
                    y=losshistory[:, i + 1 + 5],
                    mode='lines',
                    line=dict(width=1, color='blue', dash='dash'),
                    name=f'Test {loss_name}',
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"  # hover title
                        "epoch: %{x:.0f}<br>"
                        "loss: %{y:.2e}<br>"
                        "<extra></extra>"  # remove extra box
                    )
                ), row=row, col=1)
                fig.update_yaxes(title_text='Log10(Loss)', row=row, col=1)
            fig.update_xaxes(title_text='Epoch', row=5, col=1)
            #fig.update_yaxes(type="log")
            fig.update_layout(
                title='Losses',
                height=5 * 250,
                showlegend=True
            )
        fig.update_yaxes(type="log", exponentformat="power")
        fig.show()

    def plot_solution(self, u, xz, vmin=None, vmax=None, ax=None):
        r""""
        Plot solution for the regularly spaced 2D grid
        Args:
            u (torch.Tensor): tensor of solution
            xz (torch.Tensor): regularly spaced 2D tensor
            vmin (float): min value to plot
            vmax (float): max value to plot
            ax (matplotlib.axes._axes.Axes)
        """
        # Check dimensions
        assert u.dim() == 2
        assert xz.dim() == 2
        assert u.shape[1] == 1
        assert xz.shape[1] == 2
        assert u.shape[0] == xz.shape[0]

        xz = xz.cpu().detach().numpy()
        x = np.unique(xz[:, 0])
        z = np.unique(xz[:, 1])
        assert x.shape[0] * z.shape[0] == xz.shape[0], "The input coordinates xz are not regularly spaced"

        # Calculate vmin and vmax from data if not provided
        vmin = u.min().item() if vmin is None else vmin
        vmax = u.max().item() if vmax is None else vmax

        u = u.cpu().detach().numpy()
        u = u.reshape(x.shape[0], z.shape[0])

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        contour = ax.contourf(x, z, u.T, levels=np.linspace(vmin, vmax, 15), cmap="cividis", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour)
        #cbar = plt.colorbar(contour, format='%.2e')
        cbar.ax.set_title(label=r"$\phi, \mathrm{V \cdot m}$")

        ax.set_xlabel("x, m")
        ax.set_ylabel("z, m")
        ax.set_title("u(x, z)")
        ax.axis("equal")
        plt.tight_layout()

    def plot_pde_residual(self, pde_res, xz, x_s, vmax=None, ax=None):
        r""""
        Plot PDE residual for the regularly spaced 2D grid
        Args:
            pde_res (torch.Tensor): tensor of PDE residuals
            xz (torch.Tensor): regularly spaced 2D tensor
            x_s (float): x position of point source
            vmax (float): max value to plot
            ax (matplotlib.axes._axes.Axes)
        """
        # Check dimensions
        assert pde_res.dim() == 2
        assert xz.dim() == 2
        assert pde_res.shape[1] == 1
        assert xz.shape[1] == 2
        assert pde_res.shape[0] == xz.shape[0]

        xz = xz.cpu().detach().numpy()
        x = np.unique(xz[:, 0])
        z = np.unique(xz[:, 1])
        assert x.shape[0] * z.shape[0] == xz.shape[0], "The input coordinates xz are not regularly spaced"

        # Apply mask
        #xs = 0.1
        threshold = 0.01
        mask = (xz[..., 0] - x_s) ** 2 + (xz[..., 1] - 0.0) ** 2 >= threshold
        mse_pde = pde_res[mask].square().mean().item()

        #mse_pde = pde_res.square().mean().item()
        pde_res = pde_res.squeeze(1).cpu().detach().numpy()

        if vmax is None:
            vmin = np.min(pde_res)
            vmax = np.max(pde_res)
            vmax = max(np.abs(vmin), np.abs(vmax))

        pde_res = pde_res.reshape(x.shape[0], z.shape[0])

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        cplot = ax.contourf(x, z, pde_res.T, levels=np.linspace(-vmax, vmax, 15), cmap="seismic",
                              vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(cplot, format='%.2e')
        cbar.ax.set_title(label=r"$\mathcal{R}_{PDE}, A\cdot m^{-3}$")

        ax.set_xlabel("x, m")
        ax.set_ylabel("z, m")
        ax.set_title(r"$\mathcal{R}_{PDE}(x, z). MSE_{\mathcal{R}_{PDE}}=$" + f"{mse_pde:.1e}")
        ax.axis("equal")
        plt.tight_layout()

    def plot_high_pde_res(self, pde_res, xz_pred, tolerance, x_s, x_range=(-1, 1), z_range=(-1, 0), radius_subd=None, ax=None):
        r"""Plot the region with high PDE residual"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2))

        # Get MSE over all PDE residuals
        # Apply mask
        #xs = 0.1
        threshold = 0.01
        mask = (xz_pred[..., 0] - x_s) ** 2 + (xz_pred[..., 1] - 0.0) ** 2 >= threshold
        mse_pde = pde_res[mask].square().mean().item()
        #mse_pde = pde_res.square().mean().item()

        # Get points where the PDE residual >
        mask = torch.abs(pde_res) > tolerance
        mask = mask.squeeze(-1)

        ax.scatter(xz_pred[mask][..., 0].cpu().detach().numpy(), xz_pred[mask][..., 1].cpu().detach().numpy(), s=10, color='black',
                   edgecolors='none', alpha=0.5, label=f'Num: {torch.sum(mask).item()}')

        ax.legend(loc='lower left')
        ax.set_xlabel('x, m')
        ax.set_ylabel('z, m')
        ax.set_title(f"$\\mathcal{{R}}_{{PDE}} > {tolerance:.1e}$. $MSE_{{\\mathcal{{R}}_{{PDE}}}} = {mse_pde:.1e}$")

        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 0])

        ax.set_xlim(x_range)
        ax.set_ylim(z_range)

        ax.set_aspect('equal', 'box')

        if radius_subd is not None:
            # Add circles of the subdomains
            radius = 5 * 0.01
            semicircle = Arc(xy=(x_s, 0), width=2 * radius, height=2 * radius, theta1=180, theta2=360, color='blue',
                             linestyle='--')
            ax.add_patch(semicircle)

    def plot_solution_summary(self, model, data, k_val_lst, x_range=(-1, 1), z_range=(-1, 0), axis=None):
        r"""Plot solution and residuals at y=0"""
        pde_bc = PdeBc()

        # For PDE residual
        x_s_pred = data.points_s
        xzk_pred = create_xyz(x_range, z_range, (0, 0), 100, 50, 1).requires_grad_()
        xz_pred = torch.stack([xzk_pred[..., 0], xzk_pred[..., 1]], dim=1)
        sigma, sigma_x, sigma_z = get_sigma_vertical_fault(xz=xz_pred)

        # For BC residual of no flow
        xzk_pred_bc = create_xyz(x_range, (0, 0), (0, 0), 100, 1, 1).requires_grad_()
        xz_pred_bc = xzk_pred_bc[:, [0, 1]]
        x_pred_bc = xz_pred_bc[:, 0:1]
        sigma_bc = get_sigma_vertical_fault(xz=xz_pred_bc, grad=False)

        # Set figure
        if axis is None:
            fig, axis = plt.subplots(nrows=len(k_val_lst), ncols=3, squeeze=False, figsize=(3 * 6, len(k_val_lst) * 3))

        # Iterate over k values
        for k_val, ax in zip(k_val_lst, axis):
            k_tensor = torch.full((xz_pred.shape[0], 1), k_val, device=xz_pred.device)
            # concatenate: [x, z, k]
            xzk_pred = torch.cat([xz_pred[:, 0:1], xz_pred[:, 1:2], k_tensor], dim=-1)
            xz_pred = torch.stack([xzk_pred[..., 0], xzk_pred[..., 1]], dim=1)

            u_pred = model.forward(xzk=xzk_pred)
            u0_components = get_analytical_solution(x_s=x_s_pred, xzk=xzk_pred, sigma0=data.sigma0)

            pde_res_pred = pde_bc.pde(u_pred, u0_components=u0_components, sigma_components=(sigma, sigma_x, sigma_z), xzk=xzk_pred, x_s=x_s_pred, sigma0=data.sigma0)

            self.plot_solution(u_pred.squeeze(0), xz_pred.squeeze(0), ax=ax[0])
            ax[0].set_title(fr"$\phi_1(x, z). log10k={k_val:.2e}$")
            ax[0].scatter(data.points_s, 0, color='black', marker='v', s=10)

            # Plot high residual points
            tolerance = 1e-01
            self.plot_high_pde_res(pde_res_pred.squeeze(0), xz_pred.squeeze(0), tolerance, x_s=x_s_pred, x_range=x_range, z_range=z_range, ax=ax[1])
            #######################
            # Get BC residual of no flow
            k_tensor = torch.full((xz_pred_bc.shape[0], 1), k_val, device=xz_pred_bc.device)
            # concatenate: [x, z, k]
            xzk_pred_bc = torch.cat([xz_pred_bc[:, 0:1], xz_pred_bc[:, 1:2], k_tensor], dim=-1)
            xz_pred_bc = torch.stack([xzk_pred_bc[..., 0], xzk_pred_bc[..., 2]], dim=1)

            u_pred_bc = model.forward(xzk_pred_bc)
            bc_res = pde_bc.bc_no_flow(u_pred_bc, sigma=sigma_bc, sigma0=data.sigma0, xzk=xzk_pred_bc)

            # Sort by x
            idx_order = torch.argsort(x_pred_bc.squeeze(), dim=0)
            x_pred_bc = x_pred_bc[idx_order]
            x_pred_bc.cpu().detach().numpy()
            bc_res = bc_res[idx_order, :]

            bcres_vmin = bc_res.min()
            bcres_vmax = bc_res.max()
            bcres_vmax = max(bcres_vmin.abs().item(), bcres_vmax.abs().item())

            bc_res = bc_res.cpu().detach().numpy()
            mse_bc_res = np.mean(np.square(bc_res))

            ax[2].plot(x_pred_bc.cpu().detach().numpy(), bc_res, '-o', markersize=2)
            ax[2].axvline(x=x_s_pred, linestyle='--', color='green')
            ax[2].set_title(
                r"$(\sigma_0 + \sigma_1)\frac{\partial \phi_1}{\partial z}(x). MSE_{\mathcal{R}_{BC}}=$" + f"{mse_bc_res:.1e}")
            ax[2].set_xlabel('x, m')
            ax[2].set_ylabel(r'$(\sigma_0 + \sigma_1) \frac{\partial \phi}{\partial z}, \mathrm{A\cdot m^{-1}}$')
            ax[2].set_ylim(-bcres_vmax, bcres_vmax)
        plt.tight_layout()
        return fig

    def save_plot_solution_summary(self, model, data, k_val_lst, x_range, z_range, checkpoint_folder, epoch):
        fig = self.plot_solution_summary(model, data, k_val_lst, x_range, z_range)

        checkpoint_fig_folder = pathlib.Path(checkpoint_folder) / "figs"
        fig.savefig(checkpoint_fig_folder / f'model_{epoch}k.png')
        plt.close(fig)
        pass