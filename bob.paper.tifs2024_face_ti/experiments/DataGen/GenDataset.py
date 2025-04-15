# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import argparse
parser = argparse.ArgumentParser(description='Generate training dataset for face reconstruction')
parser.add_argument('--prot_scheme', metavar='<prot_scheme>', type= str, default='unprotected',
                    help='protection scheme to be used')
parser.add_argument('--same_r', metavar='<same_r>', type= str, default='none',
                    help='takes values, none, true, false')
parser.add_argument('--data_dir', metavar='<data_dir>', type= str, default='lfw_pp',
                    help='dataset to be used')
parser.add_argument('--mode', metavar='<mode>', type= str, default='test',
                    help='generate from train set or test set')
args = parser.parse_args()


from numpy.lib.type_check import imag
import torch
from torch.utils.data import Dataset

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from model import InceptionResnetV1MRP, InceptionResnetV1FE, InceptionResnetV1
# from bob.bio.face.embeddings.pytorch import IResnet100

import cv2
import os
import glob
import random
import numpy as np

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_FaceRecognition_transformer(prot_scheme='unprotected', same_r = 'none', device='cpu'):
    pretrained = 'vggface2'
    if prot_scheme== 'unprotected':
        FaceRecognition_transformer = InceptionResnetV1(pretrained = pretrained, device=device)
    elif prot_scheme== 'mrp':
        if same_r == 'true':
            FaceRecognition_transformer = InceptionResnetV1MRP(pretrained = pretrained, device=device) # true by default
        elif same_r == 'false':
            FaceRecognition_transformer = InceptionResnetV1MRP(pretrained = pretrained, device=device, sameR=False) 
    elif prot_scheme== 'fe':
        if same_r == 'true':
            FaceRecognition_transformer = InceptionResnetV1FE(pretrained = pretrained, device=device) #true by default
        elif same_r == 'false':
            FaceRecognition_transformer = InceptionResnetV1FE(pretrained = pretrained, device=device, sameR=False)
    else:
        print(f"Bad prot_scheme: {prot_scheme}")
    print(prot_scheme, same_r)
    return FaceRecognition_transformer 

class Generate_Dataset():
    def __init__(self, data_dir = 'celeba_pp',
                        mode = 'train',
                        device='cpu',
                        prot_scheme='unprotected',
                        same_r='none'
                ):
        self.device=device
        self.dir_all_images = []
        self.mode = mode
        self.prot_scheme = prot_scheme
        self.same_r = same_r
        self.data_dir = data_dir
        
        folder = os.path.join(data_dir, mode, "images")
        print(folder)
        all_imgs = glob.glob(folder+'/*') # can be png or jpg
        all_imgs.sort()
        if mode == 'test': # pick 1000 only
            all_imgs = all_imgs[:1000]
        
        for img in all_imgs:
            self.dir_all_images.append(img)
                        
        self.Face_Recognition_Network = get_FaceRecognition_transformer(prot_scheme=prot_scheme, same_r = same_r, device=self.device).eval()

    
    def generate_dataset(self, save_dir):
        os.makedirs(f'{save_dir}/images',exist_ok=True)
        os.makedirs(f'{save_dir}/embeddings',exist_ok=True)
        for idx in range(len(self.dir_all_images)):
            if (idx+1)%100 == 0:
                print(idx+1)
            embedding, image = self.__getitem__(idx)
            np.save(f'{save_dir}/images/{self.dir_all_images[idx][-10:-4]}', image)
            np.save(f'{save_dir}/embeddings/{self.dir_all_images[idx][-10:-4]}', embedding)

    def __getitem__(self, idx):

        image = cv2.imread(self.dir_all_images[idx]) # (160, 160, 3) already cropped and scaled
        # image = cv2.resize(image, (160,160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image/255.
        # print(image)
        # removed augment
        image = image.transpose(2,0,1)  # (3, 160, 160)
        image = np.expand_dims(image, axis=0) # (1, 3, 160, 160)
        
        img = torch.Tensor( image.astype(np.float32) ).type(torch.FloatTensor)

        id = int(self.dir_all_images[idx].split('/')[-1][:-4]) # remove .npy from the name

        if self.prot_scheme == 'unprotected':
            embedding = self.Face_Recognition_Network(img.to(self.device))
        else:
            embedding = self.Face_Recognition_Network(img.to(self.device), img_id = id) 
        # embedding = self.Face_Recognition_Network.transform( (image*255.).astype('uint8') )

        image = image[0] # range (0,1) and shape (3, 160, 160)

        image = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        
        return embedding, image
    
    def transform_image(self,image):
        image = image/255.
        image = image.astype(np.float32)
        image = image.transpose(1,2,0) # (160, 160, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose(2,0,1) # (3, 160, 160)
        return image
    
    def transform_embedding(self,embedding):
        embedding = embedding.detach().cpu().numpy()
        embedding = np.reshape(embedding,[embedding.shape[-1],1,1])
        return embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)


DB_gen = Generate_Dataset(data_dir = args.data_dir, mode = args.mode, prot_scheme=args.prot_scheme, same_r=args.same_r, device=device)
DB_gen.generate_dataset(save_dir=f'./databases/{args.data_dir}/{args.prot_scheme}_{args.same_r}/{args.mode}')