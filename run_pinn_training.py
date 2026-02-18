import gc
import sys
import os
import signal
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8-colorblind')

import logging
import time
from logger import setup_logger, log_grad_norms, log_gpu_mem, log_device_tensors
from models import ModelSingleElectrode, MSSiren
from losses import Losses
from losshistory import LossHistory
from visualization import Visualizer
vis = Visualizer()

from train_loop import train, get_lr_factor
from pinn_data import generate_data, select_k_vals, run_rmas, save_train_tensors, load_train_tensors, make_dataloaders, reset_iterators, get_prepared_batches
from get_sigma import _load_sigma_parameters

def set_cpu_limits(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

# def handle_sigterm(_signum, _frame):
#     logger.info("Received termination signal.")
#     logging.shutdown()  # Flush all logging handlers
#     time.sleep(0.1)  # Allow time for flushing (optional)
#     sys.exit(0)

def run(x_s, k_idx, epochs_per_train, display_every, checkpoint_folder, fpath_model=None, epoch_start=None, loss_lst_start=None, lbfgs_flag=True, adam_warmup=True, **kwargs):
    logger = setup_logger(logger_name=f"run_{checkpoint_folder.split("ver", 1)[1]}")

    logger.info("The code has been run")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device usage: {device}")

    # Generate data
    sigma_0, _, _, _= _load_sigma_parameters()

    data = generate_data(device, k_idx_range=(k_idx, k_idx), sigma_0=sigma_0, x_s=x_s, n_pde=10000, n_pde_subd=5000, n_bc=1500, distr_pde_file=None, seed=0, logger=logger)
    load_train_tensors(data, logger)

    ##############
    for pt_name in data.points.keys():
        data.points[pt_name][..., 2] = data.logk_val_lst[0].item()
    ##############

    # Generate test data
    data_test = generate_data(device, k_idx_range=(k_idx, k_idx), sigma_0=sigma_0, x_s=x_s, n_pde=5000, n_pde_subd=1000, n_bc=500, distr_pde_file=None, seed=10, logger=logger)
    ##############
    for pt_name in data.points.keys():
        data.points[pt_name][..., 2] = data.logk_val_lst[0].item()
    ##############

    data_test.batch_sizes_dict = {domain: pts.shape[0] for domain, pts in data_test.points.items()}  # Full dataset
    make_dataloaders(data_test, data_test.batch_sizes_dict, shuffle=False, pin_memory=not data_test.points['pde_domain'].is_cuda)
    reset_iterators(data_test)
    get_prepared_batches(data_test)

    # Initialize the model
    torch.manual_seed(0)
    model = ModelSingleElectrode(latent_dim=128, num_hidden_layers_tr=8, num_hidden_layers_br=8, scale_factors=(1, 2, 4, 8), use_skip=True).to(data.device)
    model.log_initialization(logger)

    # Load the model
    if fpath_model is not None:
        model = torch.load(fpath_model, weights_only=False, map_location=data.device)
        logger.info(f'The model is loaded from {fpath_model}')

    # Train
    # Set folder to save the model checkpoints
    (Path(checkpoint_folder) / "figs").mkdir(parents=True, exist_ok=True)

    # Initialize Loss history
    loss_names = ("total_loss", "loss_pde_domain", "loss_pde_subd", "loss_bc_top", "loss_robin")
    losshistory = LossHistory(loss_names, logger, track_test=True)

    k_val_lst_to_display = [data.logk_val_lst_full[i].item() for i in [k_idx,]]

    # Set the last epoch and losses
    if epoch_start is not None:
        losshistory.update(epoch=epoch_start, loss_lst=loss_lst_start)
    else:
        # Save the initial model
        Path(checkpoint_folder) / f'model_{0}k.pth'
        torch.save(model, Path(checkpoint_folder) / f'model_{0}k.pth')
        vis.save_plot_solution_summary(model, data, k_val_lst_to_display, x_range=(-2, 2), z_range=(-2, 0), checkpoint_folder=checkpoint_folder, epoch=0)

    # Define loss function: loss_pde_domain", "loss_pde_subd", "loss_bc_top", "loss_bc_robin"
    loss_fns = Losses(loss_weights=torch.tensor(2*[1] + [1] + [1], dtype=torch.float32, device=data.device))
    loss_fns.loss_weights = nn.Parameter(torch.ones(4, dtype=torch.float32, device=data.device))

    if adam_warmup:
        start_lr = 1e-6
        end_lr = 1e-3
        warmup_steps = 5000
        optimizerAdam = torch.optim.Adam(model.parameters(), lr=end_lr, weight_decay=1e-5) # optimizerAdam.param_groups[0]['lr']
        scheduler = LinearLR(optimizerAdam, start_factor=start_lr / end_lr, end_factor=1.0, total_iters=warmup_steps)
        train(losshistory, model, data, data_test=data_test, optimizer=optimizerAdam, loss_fns=loss_fns,
              epochs=warmup_steps, display_every=100, k_val_lst_to_display=k_val_lst_to_display,
              checkpoint_folder=checkpoint_folder, scheduler=scheduler, earlystop_testloss=False, earlystop_lr=False, logger=logger)

    tolerance_rmas = 5*1e-00 # 5*1e-00
    lr_adam = 1e-03
    Tmax = 3000
    base_lr = 5*1e-04
    base_Tmax = 8000

    for i in range(1000):
        logger.info(f"Pass {i}")
        data.batch_sizes_dict = {domain: pts.shape[0] for domain, pts in data.points.items()}  # Full dataset
        make_dataloaders(data, data.batch_sizes_dict, pin_memory=not data.points['pde_domain'].is_cuda)
        reset_iterators(data)
        #optimizerAdam = torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=1e-05)

        optimizerAdam = torch.optim.Adam([
            {"params": model.parameters(), "lr": lr_adam, "weight_decay": 0.0}, # "weight_decay": 1e-5
            {"params": [loss_fns.loss_weights], "lr": 1e-2, "weight_decay": 0.0},  # higher LR, no decay
        ])

        scheduler = ReduceLROnPlateau(optimizerAdam, mode='min', factor=0.5, patience=200, cooldown=100, min_lr=1e-06, threshold=1e-04, threshold_mode='rel')
        logger.info(f'Adam learning rate: {optimizerAdam.param_groups[0]['lr']:.2e}')
        train(losshistory, model, data, data_test=data_test, optimizer=optimizerAdam, loss_fns=loss_fns,
              epochs=epochs_per_train, display_every=display_every, k_val_lst_to_display=k_val_lst_to_display, checkpoint_folder=checkpoint_folder, scheduler=scheduler,
              earlystop_testloss=False, logger=logger)
        gc.collect()
        torch.cuda.empty_cache()

        # LBFGS
        if lbfgs_flag:
            logger.info(f'LBFGS optimizer:')
            data.batch_sizes_dict = {domain: pts.shape[0] for domain, pts in data.points.items()}  # Full dataset must be for LBFGS optimizer
            make_dataloaders(data, data.batch_sizes_dict, pin_memory=not data.points['pde_domain'].is_cuda)
            reset_iterators(data)
            optimizerLBFGS = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=10000, max_eval=None, tolerance_grad=1e-15, tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe")
            train(losshistory, model, data, data_test=data_test, optimizer=optimizerLBFGS, loss_fns=loss_fns,
                  epochs=200, display_every=50, k_val_lst_to_display=k_val_lst_to_display, checkpoint_folder=checkpoint_folder, scheduler=None,
                  earlystop_testloss=False, logger=logger) # from gemini: epochs=1
            gc.collect()
            torch.cuda.empty_cache()
        log_grad_norms(model, logger)

        # Adaptive sampling, add points with high PDE and BC residuals
        with rar_lock:
            while True:
                torch.cuda.empty_cache() # Clear cache to ensure maximum room
                count_bad_point_lst = run_rmas(model, data, num_lst=[500, 500, 50], tolerance=tolerance_rmas, logger=logger) # [10, 10, 3]
                torch.cuda.empty_cache() # Clear cache again so other processes can use the space

                total_bad_points = sum(count_bad_point_lst)
                save_train_tensors(data, folder_name=f"train_tensors_{checkpoint_folder.split("ver", 1)[1]}")
                if all(item <= 1 for item in count_bad_point_lst): # decrease tolerance if <= bad point per domain
                    tolerance_rmas = tolerance_rmas / 5
                    logger.info(f'New residual tolerance rmas: {tolerance_rmas}')
                    base_lr = base_lr/1.2
                else:
                    break

        # Update Adam lr and scheduler Tmax
        scale_lr = min(total_bad_points, 23) / 23
        lr_adam = 1e-5 + (base_lr - 1e-5) * scale_lr

        # Reset best_test_epoch
        losshistory.best_test_loss = np.inf

        # Save the model at the end of the pass
        torch.save(model, Path(checkpoint_folder) / f'model_{losshistory.current_epoch}k.pth')
        vis.save_plot_solution_summary(model, data, k_val_lst_to_display, x_range=(-2, 2), z_range=(-2, 0), checkpoint_folder=checkpoint_folder, epoch=losshistory.current_epoch)

        if tolerance_rmas <= 10**-3:
            logger.info("Optimization reached the solution")
            break

    logger.info("The code has been finished")

def init_worker(shared_lock):
    """The Global Initializer (Mandatory for Pool)"""
    global rar_lock
    rar_lock = shared_lock

def run_wrapper(task_dict):
    """This unpacks the dictionary keys as keyword arguments"""
    # Get number of CPUs from the shell file
    if task_dict['set_num_cores']:
        with open('run_hpc.sh', 'r') as f:
            line = f.readlines()[16]
        num_cores = int(line.strip())

        # Get number of tasks for parallel processing
        num_tasks = task_dict['num_tasks']

        set_cpu_limits(int(num_cores/num_tasks)-1)
        import torch
        torch.set_num_threads(int(num_cores/num_tasks)-1)  # Set this to (Total CPUs / Number of Processes)
    return run(**task_dict)

if __name__ == "__main__":
    # Register the SIGTERM handler
    # signal.signal(signal.SIGTERM, handle_sigterm)
    # signal.signal(signal.SIGINT, handle_sigterm)

    # MANDATORY for PyTorch + CUDA + Multiprocessing
    mp.set_start_method('spawn', force=True)

    # Create the shared lock
    lock = mp.Lock()

    x_s = 0.1
    all_tasks = [
        dict(x_s=x_s, k_idx=9, checkpoint_folder="checkpoints_model_ver25_8", fpath_model=None),
    ]
    # Parameters common for all tasks
    par_common = dict(epochs_per_train=10_000, display_every=500, lbfgs_flag=False, adam_warmup=False, set_num_cores=True, num_tasks=len(all_tasks))
    all_tasks = [{**task, **par_common} for task in all_tasks]

    with mp.Pool(processes=len(all_tasks), initializer=init_worker, initargs=(lock,)) as pool:
        pool.map(run_wrapper, all_tasks)
