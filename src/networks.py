import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def make_nets(config, overwrite, training=True):
    """Creates Generator and Discriminator class objects from params either loaded from config object or params file.

    :param config: a Config class object 
    :type config: Config
    :param training: if training is True, params are loaded from Config object. If False, params are loaded from file, defaults to True
    :type training: bool, optional
    :return: Discriminator and Generator class objects
    :rtype: Discriminator, Generator
    """
    # save/load params
    if training:
        config.save()
    else:
        config.load()

    k, s, f, p = config.get_net_params()
    
    if config.net_type == 'cnn':
        class Net(nn.Module):           # allows nn.Module functions to be used in the class
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                for lay, (ker, str, pad) in enumerate(zip(k, s, p)):
                    self.convs.append(
                        nn.Conv2d(f[lay], f[lay+1], ker, str, pad)
                    )

            def forward(self, x):
                for conv in self.convs[:-1]:
                    x = F.relu_(conv(x))
                x = self.convs[-1](x)  # bs x 1 x 1
                return F.softmax(x, dim=1)
    return Net

