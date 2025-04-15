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
parser = argparse.ArgumentParser(description='Train face reconstruction network')
parser.add_argument('--data_dir', metavar='<data_dir>', type= str, default='lfw_pp',
                    help='dataset to be used')
parser.add_argument('--prot_scheme', metavar='<prot_scheme>', type= str, default='unprotected',
                    help='protection scheme to be used')
parser.add_argument('--same_r', metavar='<same_r>', type= str, default='none',
                    help='takes values, none, true, false')
parser.add_argument('--length_of_embedding', metavar='<length_of_embedding>', type= int, default=512,
                    help='length of embedding')
args = parser.parse_args()

import os,sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset
import  bob.io.base
import glob
import random
import numpy as np
import cv2


seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


length_of_embedding=args.length_of_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)


#=================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(device=device, dataset_dir = f'{args.data_dir}', mode = 'train')
testing_dataset  = MyDataset(device=device, dataset_dir = f'{args.data_dir}', mode = 'test')

train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(testing_dataset,  batch_size=32, shuffle=False)
#========================================================
print('dataloaded')
#=================== import Network =====================
from src.Network import Generator
model_Generator = Generator(length_of_embedding=length_of_embedding)
model_Generator.to(device)
#========================================================
print('generator loaded')
#=================== import Loss ========================
# ***** VGGPerceptualLoss
from src.loss.PerceptionLoss import VGGPerceptualLoss
Perceptual_loss=VGGPerceptualLoss()
Perceptual_loss.to(device)

# ***** SSIM_Loss
from src.loss.SSIMLoss import SSIM_Loss
ssim_loss = SSIM_Loss()
ssim_loss.to(device)

# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss
ID_loss = ID_Loss(prot_scheme=args.prot_scheme,same_r=args.same_r , device=device)

# ***** Other losses
MAE_loss = torch.nn.L1Loss()
MSE_loss = torch.nn.MSELoss()
BCE_loss = torch.nn.BCELoss()
#========================================================
print('losses loaded')

#=================== Optimizers =========================
# ***** optimizer_Generator
optimizer_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-3)
scheduler_Generator = torch.optim.lr_scheduler.StepLR(optimizer_Generator, step_size=10, gamma=0.5)
#========================================================



#=================== Save models and logs ===============
import os
training_files = 'training_files/'+args.prot_scheme+'_'+args.same_r
os.makedirs(f'{training_files}',exist_ok=True)
os.makedirs(f'{training_files}/models',exist_ok=True)
os.makedirs(f'{training_files}/Generated_images',exist_ok=True)
os.makedirs(f'{training_files}/logs_train',exist_ok=True)

with open(f'{training_files}/logs_train/generator.csv','w') as f:
    f.write("epoch,MAE_loss_Gen,ID_loss_Gen,Perceptual_loss_Gen,ssim_loss_Gen_test,total_loss_Gen\n")

with open(f'{training_files}/logs_train/log.txt','w') as f:
    pass


for embedding, real_image, id in test_dataloader:
    pass
real_image=real_image.cpu()
for i in range(real_image.size(0)):
    #if i >2: 
    #    break
    os.makedirs(f'{training_files}/Generated_images/{i}', exist_ok=True)
    img = real_image[i].squeeze()
    im = (img.numpy().transpose(1,2,0)*255).astype(int)
    # cv2.imwrite(f'{training_files}/Generated_images/{i}/real_image.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
    cv2.imwrite(f'{training_files}/Generated_images/{i}/real_image.jpg',im)
#========================================================
print('starting training')
#=================== Train ==============================
num_epochs=200
for epoch in range(num_epochs):  
    iteration=0
    
    print(f'epoch: {epoch}, \t learning rate: {optimizer_Generator.param_groups[0]["lr"]}')
    model_Generator.train()
    for embedding, real_image, id in train_dataloader:
        # ==================forward==================
        generated_image = model_Generator(embedding)
        MAE = MAE_loss(generated_image, real_image)
        if args.same_r == 'false':
            ID  = ID_loss(generated_image, real_image, id)
        else:
            ID  = ID_loss(generated_image, real_image)
        Perceptual  = Perceptual_loss(generated_image, real_image)
        ssim = ssim_loss(generated_image, real_image)
        total_loss_Generato = MAE + 0.75*ssim  + 0.02*Perceptual  + 0.025*ID 
        
        # ==================backward=================
        optimizer_Generator.zero_grad()
        total_loss_Generato.backward()
        optimizer_Generator.step()
        # ==================log======================
        iteration +=1
        if iteration % 200 == 0:
            # model_Generator.eval()
            #print(f'epoch:{epoch+1} \t, iteration: {iteration}, \t total_loss:{total_loss_Generato.data.item()}')
            with open(f'{training_files}/logs_train/log.txt','a') as f:
                f.write(f'epoch:{epoch+1}, \t iteration: {iteration}, \t total_loss:{total_loss_Generato.data.item()}\n')
            pass
        
    # ******************** Eval Genrator ********************
    model_Generator.eval()
    MAE_loss_Gen_test = ID_loss_Gen_test = Perceptual_loss_Gen_test = ssim_loss_Gen_test = total_loss_Gen_test = 0
    iteration =0
    for embedding, real_image, id in test_dataloader:
        iteration +=1
        # ==================forward==================
        with torch.no_grad():
            generated_image = model_Generator(embedding)
            MAE = MAE_loss(generated_image, real_image)
            if args.same_r == 'false':
                ID  = ID_loss(generated_image, real_image, id)
            else:
                ID  = ID_loss(generated_image, real_image)
            Perceptual  = Perceptual_loss(generated_image, real_image)
            ssim = ssim_loss(generated_image, real_image)
            total_loss_Generato = MAE + 0.75*ssim  + 0.02*Perceptual  + 0.025*ID 
            ####
            MAE_loss_Gen_test += MAE.item() 
            ID_loss_Gen_test  += ID.item()
            Perceptual_loss_Gen_test += Perceptual
            ssim_loss_Gen_test  += ssim.item()
            total_loss_Gen_test += total_loss_Generato.item()

    with open(f'{training_files}/logs_train/generator.csv','a') as f:
        f.write(f"{epoch+1}, {MAE_loss_Gen_test/iteration}, {ID_loss_Gen_test/iteration}, {Perceptual_loss_Gen_test/iteration}, {ssim_loss_Gen_test/iteration}, {total_loss_Gen_test/iteration}\n")
        
    generated_image = model_Generator(embedding).detach().cpu()
    for i in range(generated_image.size(0)):
        #if i >2: 
        #    break
        img = generated_image[i].squeeze()
        im = (img.numpy().transpose(1,2,0)*255).astype(int)
        # cv2.imwrite(f'{training_files}/Generated_images/{i}/epoch_{epoch+1}.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
        cv2.imwrite(f'{training_files}/Generated_images/{i}/epoch_{epoch+1}.jpg',im)
    # *******************************************************
    
    # Save model_Generator
    torch.save(model_Generator.state_dict(), f'{training_files}/models/Generator_{epoch+1}.pth')
    
    # Update oprimizer_Generator lr
    scheduler_Generator.step()
#========================================================
