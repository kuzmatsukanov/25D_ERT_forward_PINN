import logging
import sys
import torch

def log_grad_norms(model, logger):
    high_threshold = 1.0    # gradient norms above this indicate possible instability
    low_threshold = 1e-5    # gradient norms below this indicate vanishing gradients
    param_threshold = 1e-3  # parameter norms below this indicate near-zero weights/biases

    high_grads = []
    low_grads = []
    small_params = []
    total_grad_norm = 0.0
    num_params = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            num_params += 1
            # Gradient norm
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm ** 2

            if grad_norm > high_threshold:
                high_grads.append((name, grad_norm))
            elif grad_norm < low_threshold:
                low_grads.append((name, grad_norm))

            # Parameter norm
            param_norm = param.data.norm(2).item()
            if param_norm < param_threshold:
                small_params.append((name, param_norm))

    total_grad_norm = total_grad_norm ** 0.5
    logger.info(f"Total Gradient Norm: {total_grad_norm:.3f} (across {num_params} params)")

    if high_grads:
        logger.info(f"Layers with HIGH gradient norms ({len(high_grads)}/{num_params}):")
        for name, norm in high_grads:
            logger.info(f" - {name}: grad_norm={norm:.3f}")

    if low_grads:
        logger.info(f"Layers with LOW gradient norms ({len(low_grads)}/{num_params}):")
        for name, norm in low_grads:
            logger.info(f" - {name}: grad_norm={norm:.3f}")

    if small_params:
        logger.info(f"Layers with SMALL parameter norms (<{param_threshold}) ({len(small_params)}/{num_params}):")
        for name, norm in small_params:
            logger.info(f" - {name}: param_norm={norm:.3f}")

def log_gpu_mem(prefix: str = "") -> str:
    """Return a short string with current & peak GPU memory usage (GB)."""
    gb = 1024 ** 3
    cur_alloc = torch.cuda.memory_allocated()      / gb
    cur_rsv   = torch.cuda.memory_reserved()       / gb
    peak_alloc = torch.cuda.max_memory_allocated() / gb
    return f"{prefix}alloc={cur_alloc:.2f} GB | reserved={cur_rsv:.2f} GB | peak={peak_alloc:.2f} GB"

def log_device_tensors(data, logger):
    point_attrs = [
    'points_pde_foregr',
    'points_pde_backgr',
    'points_bc_foregr',
    'points_bc_backgr',
    'points_bc_bottom',
    'points_bc_left',
    'points_bc_right'
    ]

    # Log tensors
    for attr in point_attrs:
        if hasattr(data, attr):
            obj_attr = getattr(data, attr)
            logger.info(f"{attr}: {obj_attr.device}")
        else:
            logger.warning(f"Attribute {attr} not found on data object")

    # Log cache
    for attr in point_attrs:
        cache_attr = attr + '_cache'
        if hasattr(data, cache_attr):
            cache_dict = getattr(data, cache_attr)
            if isinstance(cache_dict, dict):
                for key, value in cache_dict.items():
                    if hasattr(value, 'device'):
                        logger.info(f"{cache_attr}[{key}]: {value.device}")
                    else:
                        logger.info(f"{cache_attr}[{key}]: {value} (no device attribute)")
            else:
                logger.warning(f"{cache_attr} is not a dictionary")
        else:
            logger.warning(f"Attribute {cache_attr} not found on data object")

def setup_logger(logger_name: str, level=logging.DEBUG):
    log_file_name = f"{logger_name}.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # IMPORTANT: force your configuration (works in Jupyter/HPC too)
    logger.handlers.clear()

    if not logger.handlers:  # avoid duplicate handlers
        # Define log format (no milliseconds in timestamps)
        formatter_fh = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        formatter_ch = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # File handler (logs everything, including DEBUG)
        file_handler = logging.FileHandler(log_file_name, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter_fh)

        # Console handler (logs only INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter_ch)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # # Global exception handler
        # def log_exceptions(exc_type, exc_value, exc_traceback):
        #     logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
        #
        # sys.excepthook = log_exceptions  # Catches all unhandled exceptions
    return logger
