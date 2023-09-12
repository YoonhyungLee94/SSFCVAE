from .models import BigVGAN as Generator
import json
import os
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_vocoder(model="bigvgan_22khz_80band"):
    with open(f"bigvgan/configs/{model}.json") as f:
        data = f.read()
        
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)

    generator = Generator(h)
    state_dict_g = load_checkpoint(f"bigvgan/pretrained_models/{model}/g_05000000")
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    return generator
