import math
import pandas as pd
import torch
from evaluation import get_solution

class ERTutils:
    def __init__(self, device=None):
        self.device = device
        pass

    def get_geometric_factor(self, p1, p2, c1, c2):
        """
        Calculate geometrical factor for surface electrodes, with flat topography
        Args:
            p1, p2, c1, c2 (float): x coordinates of the correspondent electrodes
        """
        geom_factor = 2 * math.pi / (1 / abs(c1 - p1) - 1 / abs(c2 - p1) - 1 / abs(c1 - p2) + 1 / abs(c2 - p2))
        return geom_factor

    def get_xyz_for_potential_surface_electrodes(self, points_s):
        """
        Get the xyz_pred tensor for the potential surface electrodes
        Args:
            points_s (torch.Tensor): [2, 1] x-coordinates of the current electrodes
        Return:
            xyz_pred (torch.Tensor): [N, 3] with (y, z) = (0, -0.025)

        """
        # Create linspace from -1 to 1 with step 0.05
        full_tensor = torch.linspace(-1, 1, 41, device=self.device)  # 41 points to cover the range with 0.05 step

        # Filter out the x-coordinates of the current electrodes
        values_to_exclude = points_s.reshape(1, -1)  # Ensure shape [1, M] for broadcasting
        full_tensor_col = full_tensor.unsqueeze(1)  # Ensure shape [N, 1] for broadcasting

        is_close_matrix = torch.isclose(full_tensor_col, values_to_exclude)  # Shape [N, M]
        is_excluded = torch.any(is_close_matrix, dim=1)  # Shape [N]

        # Get the filtered x-coordinates as a column tensor [N, 1]
        filtered_x_col = full_tensor_col[~is_excluded]  # Shape will be [N, 1]

        # # Add (y,z) = (0,0)
        # zeros_yz = torch.zeros(filtered_x_col.shape[0], 2, device=filtered_x_col.device, dtype=filtered_x_col.dtype)
        # xyz_pred = torch.cat((filtered_x_col, zeros_yz), dim=1)  # Resulting shape [N, 3]

        # Add (y, z) = (0, -0.025)
        fixed_yz = torch.tensor([0, -0.025], device=filtered_x_col.device, dtype=filtered_x_col.dtype)
        fixed_yz = fixed_yz.repeat(filtered_x_col.shape[0], 1)
        xyz_pred = torch.cat((filtered_x_col, fixed_yz), dim=1)  # Resulting shape [N, 3]
        return xyz_pred

    def get_pinn_u_pair_c1c2(self, model, points_s):
        """
        Get PINN solution for the surface potential electrodes caused by the current injection (u_C1+ - u_C2-)
        Args:
            model: pytorch network model of u(x, y, z, x_s) = branch(x_s) * trunk(x, y, z)
            points_s (torch.Tensor): [2, 1] x-coordinates of the current electrodes
        """
        # Get PINN solution
        x_s_pred = points_s
        x_s_pred = x_s_pred[...,None]

        xz_pred = self.get_xyz_for_potential_surface_electrodes(points_s)
        xz_pred = xz_pred[None,...].expand(x_s_pred.shape[0], -1, -1)

        u_pred = get_solution(model, x_s=x_s_pred, xyz=xz_pred)['u_pred']

        # Get potentials for the pair (C1+, C2-) injection
        u_pair_pred = u_pred[1] - u_pred[0]
        df_pinn_model = pd.DataFrame({'x': xz_pred[0][:,0].cpu().detach().numpy(), 'u_pair_c1c2': u_pair_pred.squeeze(1).cpu().detach().numpy()})

        # Sort by x
        df_pinn_model = df_pinn_model.sort_values('x')
        return df_pinn_model

    def get_pinn_rhoa(self, df_pinn_u_pair_c1c2, fpath_electrodes, fpath_resipy_forward):
        """
        Get Rhoa for the surface potential electrodes
        Args:
            df_pinn_u_pair_c1c2 (pd.DataFrame): ['x', 'u_pair_c1c2']
            fpath_electrodes (str): path to csv file with the electrode coordinates ('../electrodes.csv') in Resipy
            fpath_resipy_forward (str): path to dat file with the results of forward modeling ('../resipy_folder/invdir/fwd/R2_forward.dat') in Resipy
        """
        # Get electrodes positions
        df_electrodes = pd.read_csv(fpath_electrodes)
        map_label2x = df_electrodes.set_index('label')['x'].to_dict()

        # Get the Resipy forward model result
        df_forward_model = pd.read_csv(fpath_resipy_forward, sep=r"\s+", skiprows=1, header=None,
                                       names=('C1', 'C2', 'P1', 'P2', 'Resistance', 'Rhoa'))
        df_forward_model = df_forward_model.drop(columns=['Resistance'])

        # Convert label -> x coordinate
        df_forward_model[['C1', 'C2', 'P1', 'P2']] = df_forward_model[['C1', 'C2', 'P1', 'P2']].apply(lambda col: col.map(map_label2x))
        df_forward_model[['C1', 'C2', 'P1', 'P2']] = df_forward_model[['C1', 'C2', 'P1', 'P2']].astype('float32')

        # Add potentials on P1 and P2 from the injection pair (C1+, C2-)
        df_forward_model = pd.merge_asof(
                            df_forward_model.sort_values('P1'),
                            df_pinn_u_pair_c1c2.rename(columns={'x': 'P1', 'u_pair_c1c2': 'P1_upair_pinn'}),
                            on='P1',
                            direction='nearest'
                        )

        df_forward_model = pd.merge_asof(
                            df_forward_model.sort_values('P2'),
                            df_pinn_u_pair_c1c2.rename(columns={'x': 'P2', 'u_pair_c1c2': 'P2_upair_pinn'}),
                            on='P2',
                            direction='nearest'
                        )

        # Get delta U
        df_forward_model['du_pinn'] = df_forward_model['P2_upair_pinn'] - df_forward_model['P1_upair_pinn']

        # Get geometrical factor
        df_forward_model['K'] = df_forward_model.apply(
            lambda row: self.get_geometric_factor(row['P1'], row['P2'], row['C1'], row['C2']),
            axis=1
        )

        # Get the Rhoa_pinn. (I = 1A)
        df_forward_model['Rhoa_pinn'] = df_forward_model['K'] * df_forward_model['du_pinn']
        return df_forward_model

    def get_resipy_pinn_table(self, df_pinn_rhoa):
        """
        Get the result DataFrame with Resipy and PINN Rhoa values
        Args:
            df_pinn_rhoa (pd.DataFrame) with columns ['C1', 'C2', 'P1', 'P2', 'Rhoa', 'Rhoa_pinn']
        """
        df_res = df_pinn_rhoa[['C1', 'C2', 'P1', 'P2', 'Rhoa', 'Rhoa_pinn']]
        with pd.option_context('mode.chained_assignment', None):
            df_res['Abs_diff'] = df_res['Rhoa'] - df_res['Rhoa_pinn']
        return df_res
