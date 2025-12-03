# Base dataset class for loading multimodal multidataset (MMMD)
import random
import bisect
import numpy as np
from typing import Iterable
from torch.utils.data import Dataset, ConcatDataset


class MultiModalDataset(Dataset):
    def __init__(self, data_path, transform_config, indices=None, correspondence=True):
        self.metadata_dir = data_path
        self.vision_transform = transform_config['vision']
        self.text_transform = transform_config['text']
        self.audio_transform = transform_config['audio']

        self._load_metadata()
        if indices:
            self._sub_sampling(indices, correspondence)

    def _sub_sampling(self, indices, correspondence=True):
        self.labels = self.labels[indices]
        self.keys = self.keys[indices]
        self.sentence = self.sentence[indices]
        self.im_labels = self.labels
        self.oom_labels = self.labels
        if not correspondence:
            self.oom_labels[:] = -1

    def __len__(self):
        return len(self.keys)
    
    def _get_video_item(self, video_path):
        video_data = self.vision_transform(video_path)
        return {'vision': video_data}

    def _get_audio_item(self, audio_path):
        audio_data = self.audio_transform(audio_path)
        return {'audio': audio_data}
    
    def _get_text_item(self, sentence):
        text_data = self.text_transform(sentence)
        return {'text': text_data}

    def __getitem__(self, idx):
        pass
    

class MultiDatasetDataset(ConcatDataset):
    def __init__(self, modalities, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.modalities = modalities
        self.num_datasets = len(self.datasets)
        self.dataset_modalities = [dataset.modalities for dataset in self.datasets]
        self.dataset_idxs = []
        for dataset_i, dataset in enumerate(self.datasets):
            self.dataset_idxs += [dataset_i] * dataset.__len__()

        self.place_holders = {}
        for key in self.modalities:
            for dataset in self.datasets:
                if key in dataset[0].keys():
                    self.place_holders[key] = np.zeros_like(dataset[0][key])
        for i in range(self.num_datasets):
            print(f'The {i}-th dataset contains {len(self.datasets[i])} number of examples.')

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        item = self.datasets[dataset_idx][sample_idx]
        for mod in set(self.modalities) - set(item.keys()):
            print('adding placeholder data...', end='\r')
            item.update({mod: self.place_holders[mod].copy()})
        for mod in set(item.keys()) - {'label',} - set(self.modalities):
            item.pop(mod)
        item['dataset_idx'] = dataset_idx

        return item




