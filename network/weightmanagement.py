''' weightmanagement.py
This file contains the WeightLoader and WeightSaver classes to assist with
loading/saving weights from a neural network.

'''

import torch
from model import NetworkModel

class WeightLoader:
    ''' WeightLoader class
    Used to load weight data into a neural network

    Parameters
    ----------
    weights_file : str
        The file containing the weights to load
    '''
    def __init__(self, weights_file):
        # initialize some class variables
        self.weights_file = weights_file
        self.on_byte = 0

        # read the weights file
        with open(self.weights_file, 'rb') as f:
            binary = f.read()
        
        # setup the buffer
        self.weight_buffer = torch.frombuffer(binary, dtype=torch.float32)

    def read_bytes(self, n):
        ''' Read bytes from the buffer

        Parameters
        ----------
        n : int
            The number of bytes to read from the buffer
        
        Returns
        -------
        torch.tensor
            A tensor holding the next n weights from the buffer
        '''
        self.on_byte += n
        return self.weight_buffer[self.on_byte - n:self.on_byte]

        