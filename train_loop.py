import gc
from pathlib import Path
import torch
#from logger import logger
import numpy as np

from pinn_data import get_prepared_batches
from visualization import Visualizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(losshistory, model, data, data_test, optimizer, loss_fns, epochs, display_every, k_val_lst_to_display,
          checkpoint_folder=None, scheduler=None, earlystop_testloss=True, earlystop_lr=True, logger=None):
    """
    Train a pytorch model network

    Args:
        losshistory (losshistory object)
        model (torch.nn.Module): pytorch neural network model
        data (Data object)
        data_test (Data object) for the test data set
        optimizer (torch.optim.Optimizer)
        loss_fns (object with loss functions)
        epochs (int): Number of epochs to train the model.
        display_every (int): Frequency (in epochs) at which to display and save training progress. saves the model checkpoint when a new best model is found.
        checkpoint_folder (str): Directory path to save model checkpoints.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Learning rate scheduler.
    """
    vis = Visualizer()

    # Define epoch numbering
    epoch_start, epoch_end = losshistory.get_epoch_start_end(epochs)

    # Check optimizer type to decide on closure usage
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    loss_train_hist = []
    # #state = {"all_losses": None}
    state = {}
    #state = {"all_losses": None, "last_total_loss": None}

    # --- 1. Define the shared physics calculation ---
    def calculate_physics_pass():
        """Calculates loss, backward, and clips gradients."""
        optimizer.zero_grad(set_to_none=True)

        # Calculate losses
        total_loss, all_losses_tensor = loss_fns.get_all_losses(model, data)

        # Backpropagate
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Save state for logging/scheduler
        state["all_losses"] = all_losses_tensor
        state["last_total_loss"] = total_loss.detach().item()

        return total_loss

    # --- 2. Define Closure (Only for LBFGS) ---
    def closure():
        return calculate_physics_pass()

    steps_per_epoch = int(np.max(np.array([v.shape[0] for _, v in data.points.items()]) / np.array([v for _, v in data.batch_sizes_dict.items()])))  # max(Npoints/batch_size)
    logger.info(f'Number of steps per epoch: {steps_per_epoch}')
    for epoch in range(epoch_start + 1, epoch_end + 1):
        losshistory.current_epoch = epoch

        # Training Mode
        model.train()

        for step in range(steps_per_epoch):
            get_prepared_batches(data)

            # Optimization Step
            if is_lbfgs:
                optimizer.step(closure)
            else:
                # Standard Adam step (Faster, no closure needed)
                calculate_physics_pass()
                optimizer.step()

            # Track losses
            loss_train_hist.append(state["all_losses"])

        # --- Scheduler Step (Once per epoch) ---
        if earlystop_lr:
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= 1e-06:
                logger.info(f"Early stopping triggered at epoch {epoch} epochs due to the learning rate reached 1e-06")
                return

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                model.eval()
                # use 'with torch.no_grad()' to save memory during eval
                total_loss_test = loss_fns.get_all_losses(model, data_test)[0].item()
                scheduler.step(total_loss_test)
                # scheduler.step(state["last_total_loss"])
            else:
                scheduler.step()

        if epoch % display_every == 0:
            # Calculate losses
            model.eval()
            loss_lst = (torch.sum(state["all_losses"]), state["all_losses"])
            all_losses_test = loss_fns.get_all_losses(model, data_test)[1]
            loss_lst_test = (torch.sum(all_losses_test), all_losses_test)

            # Save loss history
            losshistory.update(epoch, loss_lst, loss_lst_test)

            # Print the losses
            losshistory.print_current_losses(loss_lst, epoch, epoch_end, loss_lst_test)

            # Check if the loss changes
            try:
                if np.isclose(losshistory.last_train[-2]["total_loss"], losshistory.last_train[-1]["total_loss"], rtol=1e-5):
                    logger.info(f"Early stopping triggered at epoch {epoch} epochs due to no changing loss value")
                    return
            except IndexError:
                pass

            # Check if the test loss improves to prevent overfitting
            if earlystop_testloss:
                if epoch - losshistory.best_test_epoch > 2000:
                    logger.info(f"Early stopping triggered at epoch {epoch} epochs due to no improve in test loss for {epoch - losshistory.best_test_epoch} epochs")
                    return

            # Save the better model
            if losshistory.best_test_epoch == epoch:
                torch.save(model, Path(checkpoint_folder) / f'model_{epoch}k.pth')
                vis.save_plot_solution_summary(model, data, k_val_lst_to_display, x_range=(-2, 2), z_range=(-2, 0), checkpoint_folder=checkpoint_folder, epoch=epoch)
                pass
            gc.collect()
            torch.cuda.empty_cache()
    return

def get_lr_factor(n_added, max_added=300, max_factor=100.0):
    """
    Args:
        n_added (int): number of added bad points
    Return:
         lr_factor (float)
    """
    return 1.0 + (max_factor - 1.0) * min(n_added, max_added) / max_added
