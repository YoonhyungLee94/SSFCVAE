from .audio import *
from .logging import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Speech Enhancement')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--logdir', type=str, default='cvae')
    return parser.parse_args()

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)