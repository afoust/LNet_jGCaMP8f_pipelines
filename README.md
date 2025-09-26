# 2PiLNet jGCaMP8f Pipelines
This repository provides the official pipelines for processing jGCaMP8f light-field videos using the LISTA-based net ("2PiLnet", or just "LNet") described in this [paper](https://www.biorxiv.org/content/10.1101/2025.03.17.643718).

## Abstract
Light field microscopy (LFM) enables high throughput functional imaging by scanlessly encoding entire volumes in single snapshots.  However, LFM's computational burden and vulnerability to scattering limit its application to biological imaging. We present a light-field strategy for volumetric, scattering-mitigated neural circuit activity monitoring. A physics-based deep neural network, 2PiLnet, is trained with two-photon volumes and one-photon light fields. Light-field videos of jGCaMP8f-expressing neurons are acquired in neocortical brain slices. 2PiLnet reconstructs volumes with two-photon-like contrast and source confinement from scattered, blurry one-photon light fields from fields-of-view for which no two-photon images are provided. This enables automated segmentation and extraction of calcium signals with high signal-to-noise ratios and reduces optical crosstalk compared to conventional volume reconstruction methods. Imaging 100 volumes per second, we observe putative spikes fired at up to 10 Hz and the spatial intermingling of putative ensembles throughout 530 x 530 x 100-micron volumes. Compared to iterative algorithms, 2PiLnet workflows reduce light-field video processing times by several-fold, advancing the goal of real-time, scattering-robust volumetric neural circuit imaging for closed-loop and adaptive experimental paradigms.

## Pipeline
**1. Preprocessing** \
Usage: run `RectifyLFStack.m` in Matlab.  
Parameters used in the script are based on the optical setup outlined in the paper. 

**2. Calculate LF Activity Map** \
Usage:
```
python makeLFactivityMapFromLFvideo.py
```  
Dependencies: `jGCaMP8f_helperFuns.py`, `modelsAdv.py`, `utils.py`

**3. Finetuning Weights** \
Usage (for fine-tuning with activity map):
```
python LF2P_FineTune_ActivityMap.py --trnVol-file "2P.mat" --trnLF-file "LF1.mat" --trnLF2-file "LF2.mat" --trnLF3-file "LF3.mat" --trnLF4-file "LF4.mat" --outputs-dir "./outputs" --weights-fileFl "epochFl.pth" --weights-fileG "epochG.pth" --weights-fileD "epochD.pth" --batch-size 2 --num-epochs 10
```
Usage (for fine-tuning with LF time sequences):
```
python LF2P_FineTune_LFSeq.py --trnVol-file "2P.mat" --trnLF-file "LF1.mat" --trnLF2-file "LF2.mat" --trnLF3-file "LF3.mat" --trnLF4-file "LF4.mat" --trnLF5-file "LF5.mat" --trnLF6-file "LF6.mat" --trnLF7-file "LF7.mat" --trnLF8-file "LF8.mat" --outputs-dir "./outputs" --weights-fileFl "epochFl.pth" --weights-fileG "epochG.pth" --weights-fileD "epochD.pth" --batch-size 2 --num-epochs 100
```
Dependencies: `modelsAdv.py`, `utils.py`  

**4. Volume Reconstruction** \
Usage:
```
python LFVol_Recon.py --trnLF-file "LF.mat" --weights-fileG "epochG.pth"
```
Dependencies: `modelsAdv.py`, `utils.py`  
For volumetric reconstruction of LF video, the input to trnLF-file is the LF video frames. To reconstruct a volume of active neurons, the input is the activity map.

**5. Calcium time series extraction** \
Usage:
```
python LNet_Matrix_ROI.py
```
Dependencies: `jGCaMP8f_helperFuns.py` 
 
## Versions
Python 3  
Matlab 2024a

## Dataset
The raw light-field videos and LNet training weights are deposited at [Zenodo](https://doi.org/10.5281/zenodo.14900715)

## LNet
For training from scratch using LNet (ie. not fine-tuning weights as in Step 3 above), code for the network can be found [here](https://github.com/hverinaz/LFM-2P)

## Citation
Carmel L. Howe, Kate L.Y. Zhao, Herman Verinaz-Jadan, Pingfan Song, Samuel J. Barnes, Pier Luigi Dragotti, Amanda J. Foust
bioRxiv 2025.03.17.643718; doi: https://doi.org/10.1101/2025.03.17.643718
