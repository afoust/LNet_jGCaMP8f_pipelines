import argparse
import os
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
np.random.seed(123)

import hdf5storage
from pickle import dump,load

import torch
torch.manual_seed(123)

from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import AverageMeter, calc_psnr
from modelsAdv import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trnLF-file', type=str, required=True) # LF .mat file
    parser.add_argument('--weights-fileG', type=str)
    args= parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters
    s=3
    N_=19
    nDepths=53
    V=51
    nLayers=6
    G=InvrsModel(nIter=nLayers,nDepths=nDepths,s=s,V=V,NxN=N_*N_).to(device) # Loading trained model
    state_dict=G.state_dict()
    for n, p in torch.load(args.weights_fileG, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)

    # Loading data
    file = hdf5storage.loadmat(args.trnLF_file)
    lfTrainFTmp=file['lfTrainTmp']
    lfTrainFTmp=np.array(lfTrainFTmp).astype(np.uint8)
    lfTrainFTmp=torch.from_numpy(lfTrainFTmp).to(device)
    sz0=lfTrainFTmp.shape[0] # Dimensions of data tensor
    sz1=lfTrainFTmp.shape[1]
    sz2=lfTrainFTmp.shape[2]
    sz3=lfTrainFTmp.shape[3]
    
    volTmpSeq_=torch.zeros((sz0,nDepths,s*sz2,s*sz3),dtype=torch.uint8).to(device)# uint8 to save memory
    maxVal=torch.zeros((1,1),dtype=torch.uint8).to(device) # initialize maxVal to 0
    
    #Compute maximun value saving memory - for normalizing 
    for j in range(len(lfTrainFTmp)):
        lfTmp=lfTrainFTmp[j,:,:,:].float()
        volTmp=G(lfTmp[None,:,:,:])
        tmpMax=volTmp.max()
        if tmpMax>maxVal:
            maxVal=tmpMax

    #Save temporal stack
    for j in range(len(lfTrainFTmp)):
        lfTmp=lfTrainFTmp[j,:,:,:].float()
        volTmp=torch.nn.functional.relu(G(lfTmp[None,:,:,:],test=1)) # Applied ReLU
        volTmp[volTmp>maxVal]=maxVal # Capping range of values
        volTmp=255*volTmp/maxVal #normalization to match uint8
        volTmpSeq_[j]=volTmp
            
            
    scio.savemat('volTmpSeq_.mat', mdict={'volTmpSeq_':volTmpSeq_.cpu().numpy()})

            
            
            
  
