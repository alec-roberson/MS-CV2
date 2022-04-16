''' weightmanagement.py
This file contains the WeightLoader and WeightSaver classes to assist with
loading/saving weights from a neural network.

'''

import torch
from .model import NetworkModel

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
        self.on_byte = 5 # set the current byte to skip the header

        # load the header
        self.header = torch.frombuffer(binary, dtype=torch.int32, count=5)

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
        return self.weight_buffer[self.on_byte - n : self.on_byte]

    def load_weights(self, model:NetworkModel):
        ''' Load weights from this weight loader into the model

        Parameters
        ----------
        model : NetworkModel
            The model to load these weights into
        '''
        i = 0
        for block_dict, module in zip(model.netblocks, model.all_modules):
            if block_dict['type'] == 'convolutional':
                conv_module = module[0]
                if block_dict['batch_normalize']:
                    # BATCH NORMALIZE MODULE
                    bn_module = module[1]
                    numel = bn_module.bias.numel() # all parameters have same # of elemenets
                    # load bias
                    bn_bias = self.read_bytes(numel).view_as(bn_module.bias)
                    bn_module.bias.data.copy_(bn_bias)
                    # load weights
                    bn_weight = self.read_bytes(numel).view_as(bn_module.weight)
                    bn_module.weight.data.copy_(bn_weight)
                    # running mean
                    bn_rm = self.read_bytes(numel).view_as(bn_module.running_mean)
                    bn_module.running_mean.data.copy_(bn_rm)
                    # running var
                    bn_rv = self.read_bytes(numel).view_as(bn_module.running_var)
                    bn_module.running_var.data.copy_(bn_rv)
                else:
                    # NO BATCH NORM MEANS CONVOLUTION HAS BIAS
                    num_b = conv_module.bias.numel()
                    conv_b = self.read_bytes(num_b).view_as(conv_module.bias)
                    conv_module.bias.data.copy_(conv_b)
                # CONVOLUTION WEIGHTS
                num_w = conv_module.weight.numel()
                conv_w = self.read_bytes(num_w).view_as(conv_module.weight)
                conv_module.weight.data.copy_(conv_w)

            
            print(f"block_dict{i} = ", block_dict)
            print(f"module{i} = ", module)
            i += 1
