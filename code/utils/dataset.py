import os
import torch
from torch.utils.data.dataset import Dataset
import h5py
from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from torch import Tensor

import torch_geometric
from torch_sparse import SparseTensor, cat
from torch.utils.data.dataloader import default_collate

class WSIDataset(Dataset):
    def __init__(self, fea_dir, label_csv, preload = False):
        super(WSIDataset, self).__init__()
        self.fea_dir = fea_dir
        self.label_name = os.path.basename(label_csv).split('.')[0]
        self.csv = pd.read_csv(label_csv)
        self.slide_ids = [slide_id.split('.')[0] for slide_id in self.csv[self.csv.keys()[0]].tolist()]
        self.labels = [label for label in self.csv[self.label_name].tolist()]
        self.preload = preload

        if self.preload:
            self.patch_features = self.load_patch_features()

    def load_patch_features(self):

        patch_features = []
        for slide_id in self.slide_ids:
            f = h5py.File(os.path.join(self.fea_dir, slide_id+'.h5'))
            patch_feature = f['features']
            patch_feature = torch.as_tensor(patch_feature,dtype = torch.float32)
            patch_features.append(patch_feature)

        return patch_features

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):

        slide_id = self.slide_ids[index]
        label = self.labels[index]

        if self.label_name == 'label1':
            label = label - 1
        elif self.label_name == 'label2' and label != 0:
            label = label - 1

        label = torch.tensor(label, dtype=torch.long) 

        if self.preload:
            patch_feature = self.patch_features[index]
            return slide_id, patch_feature, label
        else:
            f = h5py.File(os.path.join(self.fea_dir, slide_id+'.h5'))
            patch_feature = f['features']
            patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)
            return slide_id, patch_feature, label

class PatchGCN_Dataset(Dataset):
    def __init__(self, fea_dir, label_csv, preload: bool = False):
        super(PatchGCN_Dataset, self).__init__()
        self.fea_dir = fea_dir
        self.label_name = os.path.basename(label_csv).split('.')[0]
        self.csv = pd.read_csv(label_csv)
        self.slide_ids = [slide_id.split('.')[0] for slide_id in self.csv[self.csv.keys()[0]].tolist()]
        self.labels = [label for label in self.csv[self.label_name].tolist()]
        self.preload = preload

        if self.preload:
            self.xs, self.adjs = self.load_patch_features()

    def load_patch_features(self):
        xs = []
        adjs = []
        for slide_id in self.slide_ids:
            data = torch.load(os.path.join(self.fea_dir, slide_id+'.pt'))
            x = torch.as_tensor(data.x, dtype = torch.float32) 
            adj = torch.as_tensor(data.edge_index, dtype = torch.int64)
            
            xs.append(x)
            adjs.append(adj)

        return xs, adjs
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):

        slide_id = self.slide_ids[index]
        label = self.labels[index]

        if self.label_name == 'label1':
            label = label - 1
        elif self.label_name == 'label2' and label != 0:
            label = label - 1

        label = torch.tensor(label, dtype=torch.long)
        if self.preload:
            x, adj = self.xs[index], self.adjs[index]
            # return slide_id, label, x, adj
            return slide_id, x, adj, label
        else:
            data = torch.load(os.path.join(self.fea_dir, slide_id+'.pt'))
            x = torch.as_tensor(data.x, dtype=torch.float32)
            adj = torch.as_tensor(data.edge_index,dtype=torch.int64)
  
            return slide_id, x, adj, label
        


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = WSIDataset('...', 
                         '...', preload=False)
    print(len(dataset))
    slide_id, patch_feature, label = dataset[0]
    patch_feature.to(device)
    print(slide_id, patch_feature.shape)
    