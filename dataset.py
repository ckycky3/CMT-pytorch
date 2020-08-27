import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ChordMusicDataset(Dataset):
    def __init__(self, root_path, frame_per_bar=16, num_bars=8, mode='train'):
        super(ChordMusicDataset, self).__init__()

        self.root_path = root_path
        self.frame_per_bar = frame_per_bar
        self.num_bars = num_bars

        self.mode = mode
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        return sorted(glob.glob(os.path.join(self.root_path, self.mode, '*/*.pkl')))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'rb') as f:
            instance = pickle.load(f)

        instance['chord'] = instance['chord'].toarray()

        return instance


def collate_fn(batch):
    result = dict()
    for key in batch[0].keys():
        content = np.array([item[key] for item in batch])
        content = torch.tensor(content)
        result[key] = content
    return result


def get_loader(config, mode='train'):
    dataset = ChordMusicDataset(root_path=config['path'], mode=mode)
    loader = DataLoader(dataset, collate_fn=collate_fn,
                        batch_size=config['loader']['batch_size'],
                        shuffle=True, drop_last=True)

    return loader