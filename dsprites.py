from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np

import torch
# Load dataset
_path = './data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

class DSPRITES(torch.utils.data.Dataset):
    
    def __init__(self, path=_path, transform=None):
        super(DSPRITES, self).__init__()
        # use naming conventions from https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
        dataset_zip = np.load(path)
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_sizes = np.array([ 1,  3,  6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
        self.transform = transform
        
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.transform(self.imgs[idx]), self.latents_values[idx]
    
    def latent_to_index(self, latents):
        return (latents @ self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples
    
    def sample_latent_dm(self, size=1):
        # choose a latent to hold constant
        const_lat = np.random.randint(2, 6)
        # randomly sample two batches of latents
        samples1 = self.sample_latent(size)
        samples2 = self.sample_latent(size)
        # iterate through all the samples and sample a new latent value for the constant lat
        for i in range(size):
            lat_val_ind = np.random.randint(0, self.latents_sizes[const_lat])
            samples1[i][const_lat] = lat_val_ind
            samples2[i][const_lat] = lat_val_ind
        # index into the dataset and return both batches
        data1 = self.__getitem__(self.latent_to_index(samples1))
        data2 = self.__getitem__(self.latent_to_index(samples2))
        return const_lat, data1, data2