import os
import torch
from imagebind.models.imagebind_model import ImageBindModel

modality_space = [
    'vision', # Vi
    'text', # Te
    'audio', # Au
    'depth', # De
    'thermal', # Th
    'imu', # Im
    'tactile', # Ta
    'point', # Po
]

class SubImageBindModel(ImageBindModel):
    ''' 
    Supported modalities: 
        'vision', 'text', 'audio', 'thermal', 'depth', 'imu'
    '''
    def __init__(self, modalities, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sup_modalities = ['vision', 'text', 'audio', 'thermal', 'depth', 'imu']
        redundant_modalities = list(set(sup_modalities) - set(modalities))

        for mod in sup_modalities:
            if mod in redundant_modalities:
                del self.modality_preprocessors[mod]
                del self.modality_trunks[mod]
                del self.modality_heads[mod]
                del self.modality_postprocessors[mod]

        for param in self.parameters():
            param.requires_grad = False

        torch.cuda.empty_cache()


def load_imagebind(args, modalities):
    model = SubImageBindModel(
        modalities,
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if not os.path.exists("./imagebind_huge.pth"):
        print(
            "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
        )
        os.makedirs("./checkpoints", exist_ok=True)
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
            "./imagebind_huge.pth",
            progress=True,
        )
        
    model.load_state_dict(torch.load("./imagebind_huge.pth"), strict=False)
    return model

