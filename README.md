# LNet jGCaMP8f Pipelines
This repository provides the official pipelines for processing jGCaMP8f light-field videos using the LISTA-based net ("LNet") described in this [paper](https://www.biorxiv.org/content/10.1101/2025.03.17.643718v1).

## Abstract
Light field microscopy enables volumetric, high throughput functional imaging. However, the computational burden and vulnerability to scattering limit light field's application to neuroscience. We present a strategy for volumetric, scattering-mitigated neural circuit activity monitoring. A physics-based deep neural network, LNet, is trained with two-photon volumes and one-photon light fields. A processing pipeline uses LNet to extract calcium activity from light-field videos of jGCaMP8f-expressing neurons in acute cortical slices. The extracted time series have high signal-to-noise ratios and reduced optical crosstalk compared to conventional volume reconstruction. Imaging 100 volumes per second, we observed putative spikes fired at up to 10 Hz and the spatial intermingling of putative ensembles throughout 530 x 530 x 100-micron volumes. Compared to iterative algorithms, LNet LFM cuts light-field video processing time from hours to minutes and hence advances the goal of real-time, scattering-robust volumetric neural circuit imaging for closed-loop and adaptive experimental paradigms.

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
Usage:
```
python LF2P_FineTune.py --trnVol-file "2P.mat" --trnLF-file "LF1.mat" --trnLF2-file "LF2.mat" --trnLF3-file "LF3.mat" --trnLF4-file "LF4.mat" --outputs-dir "./outputs" --weights-fileFl "epochFl.pth" --weights-fileG "epochG.pth" --weights-fileD "epochD.pth" --batch-size 2 --num-epochs 10
```
Dependencies: `modelsAdv.py`, `utils.py`  
For fine-tuning with LF videos, all 4 LF file inputs are LF video files. For fine-tuning with activity maps, the input to `trnLF4-file` is the activity map(s). 

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
To be uploaded on Zenodo

Network weights (not refined) are also available on Zenodo. 

## LNet
For training from scratch using LNet (ie. not fine-tuning weights as in Step 3 above), code for the network can be found [here](https://github.com/hverinaz/LFM-2P)

## Citation
Carmel L. Howe, Kate L.Y. Zhao, Herman Verinaz-Jadan, Pingfan Song, Samuel J. Barnes, Pier Luigi Dragotti, Amanda J. Foust
bioRxiv 2025.03.17.643718; doi: https://doi.org/10.1101/2025.03.17.643718
