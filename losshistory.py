#from logger import logger
import numpy as np
import re


from collections import deque
class LossHistory:
    def __init__(self, loss_names, logger, track_test=False):
        """
        Initialize LossHistory to store only the two most recent losses.
        Args:
            loss_names (iterable): names of losses to track (must start with "total_loss")
            track_test (bool): whether to also record test losses
        """
        assert loss_names[0] == "total_loss", 'loss_names[0] should be "total_loss"'

        self.logger = logger

        self.current_epoch = None

        self.loss_names = list(loss_names)
        self.track_test = track_test

        # keep only last two records
        self.last_train = deque(maxlen=2)
        if self.track_test:
            self.last_test = deque(maxlen=2)

        # best‐model tracking
        self.best_epoch = 0
        self.best_total_loss = np.inf

        if self.track_test:
            self.best_test_epoch = 0
            self.best_test_loss = np.inf

    def update(self, epoch, loss_lst, test_loss_lst=None):
        # --- train ---
        total = loss_lst[0].item()
        indiv = loss_lst[1]
        rec = {"epoch": epoch, "total_loss": total}
        for name, val in zip(self.loss_names[1:], indiv):
            rec[name] = val.item()
        self.last_train.append(rec)

        if total < self.best_total_loss:
            self.best_total_loss = total
            self.best_epoch = epoch

        # --- optional test ---
        if self.track_test and test_loss_lst is not None:
            test_total = test_loss_lst[0].item()
            test_indiv = test_loss_lst[1]
            rec_test = {"epoch": epoch, "total_loss": test_total}
            for name, val in zip(self.loss_names[1:], test_indiv):
                rec_test[name] = val.item()
            self.last_test.append(rec_test)

            if test_total < self.best_test_loss:
                self.best_test_loss = test_total
                self.best_test_epoch = epoch

    def get_epoch_start_end(self, epochs):
        try:
            start = self.last_train[-1]["epoch"] + 1
        except IndexError:
            start = 0
        return start, start + epochs

    def print_current_losses(self, loss_lst, epoch, epoch_end, loss_lst_test=None):
        total = loss_lst[0].item()
        indiv = loss_lst[1]
        loss_str = f"{total:.2e}, " + ", ".join(f"{l.item():.2e}" for l in indiv)

        if loss_lst_test is not None:
            t_total = loss_lst_test[0].item()
            t_indiv = loss_lst_test[1]
            loss_str += " | " + f"{t_total:.2e}, " + ", ".join(f"{l.item():.2e}" for l in t_indiv)

        self.logger.info(f"Epoch {epoch}/{epoch_end}, Loss: {loss_str}")


def get_losshistory_from_log_file(log_file_path, log_val=True):
    """
    Get loss history from the log file
    Args:
        log_file_path (str): path to the log file
    Return:
        numpy.ndarray: [N, 11] [epoch, train_losses, test_losses]
    """
    pattern = re.compile(r'^20.*?Epoch\s+(\d+)/.*?Loss:\s*(.*)')
    losshistory = []

    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                # Replace "|" with "," then split and convert to float
                losses = [float(x) for x in match.group(2).replace('|', ',').split(',') if x.strip()]
                losshistory.append([epoch] + losses)
    losshistory = np.array(losshistory) if losshistory else np.array([])
    if log_val:
        try:
            losshistory[:, 2:8] = np.log(losshistory[:, 2:8])
            losshistory[:, 9:15] = np.log(losshistory[:, 9:15])
        except IndexError:
            pass
    return losshistory
