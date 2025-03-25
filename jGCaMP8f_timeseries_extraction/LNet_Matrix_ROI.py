import hdf5storage
import pyclesperanto_prototype as cle
import numpy as np
import jGCaMP8f_helperFuns as hpf

# Replace with path and name of activation volumes reconstructed by LNet
fname = r'/Volumes/home/jGCaMP8f_analysis/activeCellsVolumes/net/volFrames1-500.mat'
matfile = hdf5storage.loadmat(fname)
activeCellsVols = matfile['volTmpSeq_']

# If activeCellsVols contains n activity maps, activeCellsVols.shape = (n, z, x, y). Set n equal to the activity map to be processed.
vol = activeCellsVols[3,:,:,:]

# Segment the activity map volume using Voronoi-Otsu labeling
seg = hpf.segment_vol( vol, spot_sigma_input=3 )

# show activation volume and segmented labels
(viewer,layer) = hpf.show_volAndlabels( vol, cle.pull(seg), namestr='LN')

# Replace with path and name of rectified light-field video corresponding to the activation map contained in "vol"
fname = r'/Volumes/home/jGCaMP8f_analysis/rectifiedData/lfTmps1a2d1_no4AP_1.mat'
matfile = hdf5storage.loadmat(fname)
LF = matfile['lfTrainTmp']

# Calculate light-field footprints "N" and extract each neuron's time series "T" via spatio-temporal factorization
(T,N) = hpf.extract_timeSeries( seg, vol, LF)

# Plot the time series
hpf.plot_timeSeries( T, layer )

# Replace with path and name of the LNet-reconstructed volume series corresponding to the activation map contained in "vol"
fname = r'/Volumes/home/jGCaMP8f_analysis/outputNet/outputsFineTuning/volTmpSeqs1a2d1_2.mat'
matfile = hdf5storage.loadmat(fname)
volsLN = matfile['volTmpSeq_']

# Extract the neuronal time series "TlnROI" using the segmented activity map and the LNet-reconstructed volume series
TlnROI = hpf.extract_time_series_from_vols( volsLN, cle.pull(seg) )

# Plot the time series
hpf.plot_timeSeries( TlnROI, layer )

# Replace with path and name of file for saving the time series obtained with
# LNet Matrix
np.save(r'/Volumes/home/jGCaMP8f_analysis/activeCellsVolumes/T_LNmat_fr0to499_A4', T)
# LNet ROI
np.save(r'/Volumes/home/jGCaMP8f_analysis/activeCellsVolumes/T_LNroi_fr0to499_A4', TlnROI)