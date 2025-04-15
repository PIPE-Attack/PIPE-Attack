# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math
import sys
sys.path.append('/home/ubuntu/FBI-Attack')
# print(sys.path)
from facenet_pytorch import InceptionResnetV1
import pandas as pd

id_map_csv = pd.read_csv('/home/ubuntu/FBI-Attack/bob.paper.tifs2024_face_ti-master/experiments/DataGen/celeba_pp/id_map.csv')
id_map = dict(zip(id_map_csv['img'], id_map_csv['id']))

def get_seed(img_id):
    if img_id is None:
        return 1740470466
    else:
        return id_map[img_id]

  
class FETransform(nn.Module):
    """A layer that applies a batch multiplication of A matrices, and addition of b vectors, to a batch of input feature vectors.
    All parameters are torch tensors of type float32"""
    def __init__(self, indim=512, outdim=128, q=None, device="cuda"):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.q = q
        self.device = device

        eig_vecs_path = "/home/ubuntu/FBI-Attack/comp-FOWH/eig_vecs_data_facenet.npy"
        eig_vals_path = "/home/ubuntu/FBI-Attack/comp-FOWH/eig_vals_data_facenet.npy"
        eig_vecs = torch.from_numpy(np.load(eig_vecs_path)).to(torch.float32).to(self.device)
        eig_vals = torch.from_numpy(np.load(eig_vals_path)).to(torch.float32).to(self.device)
        self.register_buffer("eig_vecs", eig_vecs)
        self.register_buffer("eig_vals", eig_vals)

        self.k = (self.eig_vals > 1).sum().item()
        
    def zq_A(self, q, m, n):
        A = np.random.rand(m, n)
        A = np.floor(A * q).astype(int) % q
        return torch.from_numpy(A).to(torch.float32)

    def zq_b(self, q, n):
        b = np.random.rand(n)
        b = np.floor(b * q).astype(int) % q
        return torch.from_numpy(b).to(torch.float32)
    
    def zq_x(self, x, q, max_proj=20, min_proj=-20):
    # almost all entries of x are from min_proj to max_proj
        x = (x - min_proj) / (max_proj - min_proj) * q
        x = (x.floor().int() % q)
        return x.to(torch.float32)
    
    def eig_x(self, x):
        x_proj = torch.matmul(x, self.eig_vecs.type_as(x))
        x_proj[:, self.k:] = 0  # Set insignificant values (eig val < 1) to 0
        return x_proj.to(torch.float32)
    
    def forward(self, x, img_id=None):
        """
        x: Tensor of shape (bs, 512)  - Feature vectors
        A: Tensor of shape (bs, 512, 128) - Transformation matrices
        b: Tensor of shape (bs, 128) - Translation vectors
        Returns:
            A*b + x mod q, Transformed tensor of shape (1, 512)
        """
        if img_id is None:
            seeds = [get_seed(None) for i in range(x.shape[0])]
        else:
            seeds = [get_seed(int(img_id[i])) for i in range(img_id.shape[0])]
        As = []
        bs = []
        for i in range(len(seeds)):
            seed = seeds[i]
            np.random.seed(seed)
            A = np.random.rand(self.indim, self.outdim)
            A = np.floor(A * self.q).astype(int) % self.q
            b = np.random.rand(self.outdim)
            b = np.floor(b * self.q).astype(int) % self.q
            # A = self.zq_A(self.q, self.indim, self.outdim)
            # b = self.zq_b(self.q, self.outdim)
            As.append(A)
            bs.append(b)
        As = np.array(As)
        bs = np.array(bs)
        # print(As[0,:5,:5])
        As = torch.from_numpy(As).to(torch.float32).to(self.device)
        bs = torch.from_numpy(bs).to(torch.float32).to(self.device)

        x = x.to(torch.float32)
        x = self.eig_x(x)
        x = self.zq_x(x, self.q) 
        c = (torch.bmm(As.type_as(x), bs.type_as(x).unsqueeze(-1)).squeeze(-1) % self.q + x) % self.q
        c = c.to(torch.float) / self.q
        return c
        
  
class InceptionResnetV1FE(InceptionResnetV1):
    def __init__(self, sameR = True, knownR = False, mode=None, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__(pretrained, classify, num_classes, dropout_prob, device)
        # Parameters for MRP embeddings
        self.sameR = sameR
        self.knownR = knownR
        self.indim = 512
        self.outdim = 128
        self.q = 130003
        self.scale = torch.sqrt(torch.tensor(self.outdim))
        self.mode = mode
        # Classification layer remains 512-dim

        self.fe = FETransform(indim=self.indim, outdim=self.outdim, q=self.q, device=device)

        if device is not None:
            self.to(self.device)

    def forward(self, x, img_id=None):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        # Size of x: (1, 512)
        x = self.fe(x, img_id)
        # Size of x: (1, 512)
        if self.classify:
            x = self.logits(x)
        return x
    
class RTransform(nn.Module):
    """A layer that applies a batch of R matrices to a batch of input feature vectors."""
    def __init__(self, indim=512, outdim=128, device='cuda'):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.scale = torch.sqrt(torch.tensor(self.outdim))
        self.device = device

    def forward(self, x, img_id=None):
        """
        x: Tensor of shape (1, 512)  - Feature vectors
        R: Tensor of shape (1, 128, 512) - Transformation matrices
        Returns:
            Transformed tensor of shape (1, 128)
        """
        # print(img_id)
        x = x.to(torch.float32)
        if img_id is None:
            seeds = [get_seed(None) for i in range(x.shape[0])]
        else:
            seeds = [get_seed(int(img_id[i])) for i in range(img_id.shape[0])]
        Rs = []
        for i in range(len(seeds)):
            seed = seeds[i]
            np.random.seed(seed)
            R = np.random.rand(self.outdim, self.indim)
            Rs.append(R)
        Rs = np.array(Rs)
        # print(Rs[0,:5,:5])
        Rs = torch.from_numpy(Rs).to(torch.float32).to(self.device)
        v = torch.bmm(Rs.type_as(x), x.unsqueeze(-1)).squeeze(-1)
        v = v / self.scale

        return v

    
class InceptionResnetV1MRP(InceptionResnetV1):
    def __init__(self, sameR = True, knownR = False, mode=None, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__(pretrained, classify, num_classes, dropout_prob, device)
        # Parameters for MRP embeddings
        self.sameR = sameR
        self.knownR = knownR
        self.mode = mode
        #* Modify classification layer for 128-dim embeddings (for unknown R)
        if self.classify and self.num_classes is not None:
            if not self.knownR:
                self.logits = nn.Linear(128, self.num_classes)

        # Use old style of MRP for backwards compatibility
        self.mrp = RTransform(device=device)

        if self.classify:
            self.logits = nn.Linear(self.outdim, self.num_classes)

        if device is not None:
            self.to(self.device)

    def forward(self, x, img_id=None):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        # Size of x: (1, 512)
                
        x = self.mrp(x, img_id)
        # Size of x: (1, 128)
        if self.classify:
            x = self.logits(x)
        return x

    