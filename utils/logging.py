import os
from torch.utils.tensorboard import SummaryWriter
import torch


def get_writer(output_directory, log_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    logging_path=f'{output_directory}/{log_directory}'
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
        
    writer = SummaryWriter(logging_path)
    return writer


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')
