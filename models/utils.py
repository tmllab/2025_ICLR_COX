import os
import torch
import logging
import torch.nn as nn
from torchmetrics import Metric

    
    
def save_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./checkpoints/", postfix: str = "_last",
                extension: str = "pth"):
    try:
        torch.save(module_dict.state_dict(),
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
        logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not save module parameters for {module_name} to {checkpoint_dir}.")


def load_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./checkpoints/", postfix: str = "_last",
                extension: str = "pth"):
    try:
        module_dict.load_state_dict(torch.load(
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")), strict=False)
        logging.info(f"Loaded parameters for module {module_name} from {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}.")
