import os
import sys
import pdb

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_handler import *

def visualize_attention(input_string, encoder, decoder, idx_dict, opts, save='save.pdf'):
    """Generates a heatmap to show where attention is focused in each decoder step.
    """

    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']
    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']

    max_generated_chars = 20
    gen_string = ''

    all_attention_weights = []

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = Variable(torch.LongTensor(indexes).unsqueeze(0))  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_hidden = encoder(indexes)

    decoder_hidden = encoder_hidden
    decoder_input = Variable(torch.LongTensor([[start_token]]))  # For BS = 1
    decoder_cell = Variable(torch.zeros(decoder_hidden.size()))

    produced_end_token = False

    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, decoder_cell, attention_weights = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_annotations)

        ni = F.softmax(decoder_output, dim=1).data.max(1)[1]  # LongTensor of size 1
        ni = ni.item()

        all_attention_weights.append(attention_weights.squeeze().data.cpu().numpy())

        if ni == end_token:
            produced_end_token = True
            break
        else:
            gen_string += index_to_char[ni]
            decoder_input = Variable(torch.LongTensor([[ni]]))

    attention_weights_matrix = np.stack(all_attention_weights)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_weights_matrix.T, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + list(input_string) + ['EOS'], rotation=90)
    ax.set_xticklabels([''] + list(gen_string) + (['EOS'] if produced_end_token else []))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save)

    plt.close(fig)

    return gen_string