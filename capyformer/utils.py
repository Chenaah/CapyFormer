


import os

import torch


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cuda"):
    """
    Loads a model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load the state.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to.
    
    Returns:
        tuple: The loaded model, optimizer, scheduler, and the epoch number.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return model, optimizer, scheduler, start_epoch


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    """
    Saves a model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save the state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save the state.
        epoch (int): The current epoch.
        checkpoint_path (str): Path to save the checkpoint.
    """
    print(f"Saving checkpoint at epoch {epoch} to {checkpoint_path}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
