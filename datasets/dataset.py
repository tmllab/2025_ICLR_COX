import os
import numpy as np
from collections import Counter
from datasets.msrvtt import MSRVTT
from imagebind import data
from datasets.base_mmmd_dataset import MultiDatasetDataset
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

def get_transform(model_type, modalities, model=None):
    transforms = find_transforms[model_type](modalities, model)
    return transforms


def get_imagebind_transform(modalities, model=None):
    transform_dict = {
        'vision': data.load_and_transform_video_data, # or data.load_and_transform_vision_data,
        'text': data.load_and_transform_text,
        'audio': data.load_and_transform_audio_data,
        'thermal': data.load_and_transform_vision_data,
        'depth': data.load_and_transform_vision_data,
        'imu': data.load_and_transform_vision_data, 
    }
    transforms = {}
    for mod in modalities:
        transforms[mod] = transform_dict.pop(mod)
    return transforms


find_dataset = {
    'msrvtt': MSRVTT,
}


find_transforms = {
    'imagebind': get_imagebind_transform,
}


def get_dataloaders(args, dataset_name, model_type, modalities, model=None):
    transform_config = get_transform(model_type, modalities, model)
    dataloaders = {}
    for split in ['val', 'test']:
        dataset = find_dataset[dataset_name](os.path.join(args.data_path, dataset_name), transform_config, split=split)
        dataset_size = dataset.__len__()
        del dataset
        labeled_size = int(dataset_size * args.rate)
        labeled_indeces = np.random.choice(dataset_size, labeled_size, replace=False)
        unlabeled_indeces = np.setdiff1d(np.arange(dataset_size), labeled_indeces)
        labeled_dataset = find_dataset[dataset_name](os.path.join(args.data_path, dataset_name), transform_config, split=split, indices=labeled_indeces)
        unlabeled_dataset = find_dataset[dataset_name](os.path.join(args.data_path, dataset_name), transform_config, split=split, indices=unlabeled_indeces, correspondence=False)
        
        mmmd_dataset = MultiDatasetDataset(modalities, [labeled_dataset, unlabeled_dataset])
        dataset_counts = Counter(mmmd_dataset.dataset_idxs)
        weights = [1.0 / dataset_counts[idx] for idx in mmmd_dataset.dataset_idxs]
        num_samples = mmmd_dataset.__len__()
        weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

        dataloaders[split] = DataLoader(mmmd_dataset, sampler=weighted_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
    return dataloaders

