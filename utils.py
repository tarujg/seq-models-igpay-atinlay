"""Utility functions.
"""

import os
import sys
import pickle as pkl

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def store_checkpoints(encoder, decoder, idx_dict, opts):
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """
    #print("Saving Model in {}".format(opts.checkpoints_dir))
    torch.save(encoder,"{}/encoder.pth".format(opts.checkpoints_dir))
    torch.save(decoder,"{}/decoder.pth".format(opts.checkpoints_dir))

def store_loss_plots(train_losses, val_losses, opts):
    """Saves a plot of the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['train', 'val'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoints_dir, 'loss_plot.pdf'))
    plt.close()

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
