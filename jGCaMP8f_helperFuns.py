#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:34:36 2024

@author: starz
"""


from scipy.ndimage import uniform_filter

import os
import glob
import torch
import scipy
import random
import scipy.io
import argparse
import skimage.io
import hdf5storage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyclesperanto_prototype as cle
import torch.optim as optim
import torch.backends.cudnn as cudnn
import napari
import sys

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from skimage import filters, morphology, measure, segmentation
from pickle import dump,load
from torch import nn
from tqdm import tqdm
from random import randint
import seaborn as sns
from scipy.fft import fft, ifft
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import binary_dilation

from utils import AverageMeter, calc_psnr
from modelsAdv import *

def thresholdBySNR_timeSeries_centroids( snrDF, time_series, centroids, SNRthresh = 4 ):
    # Example 1D array with condition
    condition_array = snrDF['SNR']  # This will filter by indexes where values are < 2
    
    # Get the indexes where the condition holds
    indexes = np.where(condition_array > SNRthresh)[0]

    return time_series[indexes, :], centroids[indexes, :], indexes


def calc_active_cells_LF( LFvid, filter_size = 3):
    S = LFvid

    #filter each pixel in time axis
    Sfilt = ndi.uniform_filter(S, size=3, axes = (0))

    #calculate the mininum value of each pixel (in time)
    smin = Sfilt.min(axis=0)

    #subtract the minimum value
    df = Sfilt - smin

    #calulate the maximum delta F for each pixel
    df_max = np.max(df, axis = 0 )

    #calcule the standard deviation for each pixel (across time)
    Sstd = np.std(S,axis=0)

    #calculate max(df)/std(F)
    dfmax_std = df_max / Sstd

    return dfmax_std

def calc_sig2noise_timeSeries( timeSeries, filterSize = 10 ):
    num_traces = timeSeries.shape[0]
    sigs = []
    noises = []
    pp = []  # To store peak-to-peak signal-to-noise ratio for each trace

    # Loop over each trace
    for k in range(num_traces):
        trace = timeSeries[k, :]
        
        # Z-score the trace
        trace_zscored = zscore(trace)
        
        # Smooth the trace using a moving average (window size 10)
        trace_smoothed = uniform_filter1d(trace_zscored, size=filterSize)
        
        # Calculate noise: standard deviation of (smoothed - original trace)
        noise = np.std(trace_smoothed - trace_zscored)
        
        # Calculate peak-to-peak signal and SNR
        signal_pp = np.max(trace_smoothed) - np.min(trace_smoothed)
        sigs.append( signal_pp )
        noises.append( noise )
        pp.append(signal_pp / noise)

    data = {'dF': np.array(sigs),
            'Noise (std)': np.array(noises),
            'SNR' : np.array(pp) }

    dataFrame = pd.DataFrame(data)
    
    return dataFrame
    
    
def calc_baseline(arr, percent = 15):
    # Calculate the size of the subarray (10% of the total length)
    subarray_size = int(np.ceil(percent/100 * len(arr)))
    
    # Initialize the minimum mean to a large value
    min_mean = float('inf')
    std = 0
    
    # Iterate over each possible subarray of the specified size
    for i in range(len(arr) - subarray_size + 1):
        # Calculate the mean of the current subarray
        current_mean = np.mean(arr[i:i + subarray_size])
        
        # Update the minimum mean if the current mean is lower
        if current_mean < min_mean:
            min_mean = current_mean
            std = np.std(arr[i:i + subarray_size])
    
    return (min_mean, std)   

def rescale_array(arr, satVal=255, out_max = 150, out_min = 9, out_median = 16.):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # Scale values between 0 and 1
    scaled_arr = scaled_arr * satVal  # Rescale values to range between 16 and the saturation value (default 255 for 8-bit range)
    arr_median = np.median(scaled_arr)
    scaled_arr = scaled_arr - (arr_median - out_median)
    scaled_arr[ scaled_arr>out_max ] = out_max
    scaled_arr[ scaled_arr<out_min ] = out_min
    return scaled_arr.astype(np.uint8)  

def scale_cast_uint8( input_array ):
    return (input_array / input_array.max() * 255.).round().astype(np.uint8)

def convert_SIDcentroids2ours( SIDcentroids ):
    centroids_valid = SIDcentroids

    centroids_ours = np.zeros(centroids_valid.shape)
    centroids_ours[:,2] = centroids_valid[:,2]
    centroids_ours[:,0] = 3 * centroids_valid[:,0] / 19 + 3
    centroids_ours[:,1] = 3 * centroids_valid[:,1] / 19 + 3

    centroids_ours_ourOrder = centroids_ours[:, [2, 0, 1]]

    lat_microns_per_voxel = 533.3 / 321
    ax_microns_per_voxel = 104 / 53
    
    centroids = centroids_ours_ourOrder
    
    centroidsSID_um = np.zeros(centroids.shape)
    centroidsSID_um[:,0] = centroids[:,0] * ax_microns_per_voxel
    centroidsSID_um[:,1] = centroids[:,1] * lat_microns_per_voxel
    centroidsSID_um[:,2] = centroids[:,2] * lat_microns_per_voxel
    
    return centroidsSID_um, centroids_ours_ourOrder

def get_centroids( segmented, lat_microns_per_voxel = 533.3 / 321, ax_microns_per_voxel = 104 / 53 ):
    regions = measure.regionprops(cle.pull(segmented))
    centroids = []
    for props in regions:
        z0, y0, x0 = props.centroid
        centroids.append([z0,y0,x0])
    centroids = np.array(centroids)
    centroids_um = np.zeros(centroids.shape)
    centroids_um[:,0] = centroids[:,0] * ax_microns_per_voxel
    centroids_um[:,1] = centroids[:,1] * lat_microns_per_voxel
    centroids_um[:,2] = centroids[:,2] * lat_microns_per_voxel

    return centroids_um

def plot_correlation_vs_distance(points, time_series):
    """
    Plot the Pearson correlation coefficient as a function of the Euclidean distance between points.

    Parameters:
    points (numpy array): Array of shape (n_points, 3) containing points in 3D space.
    time_series (numpy array): Array of shape (n_points, n_time_points) containing time series for each point.
    """
    # Calculate pairwise distances
    distances = scipy.spatial.distance.cdist(points, points)

    # Calculate pairwise Pearson correlation coefficients
    n_points = time_series.shape[0]
    correlations = np.zeros((n_points, n_points))
    z_avg = np.zeros((n_points, n_points))
    xy_dist = np.zeros((n_points, n_points))

    # filter and normalize the time series
    T = np.zeros(time_series.shape)
    
    for i in range(time_series.shape[0]):
        sig = ndi.gaussian_filter1d(time_series[i,:],sigma = 2, axis = 0) 
        sig_norm = (sig-sig.min())/(sig.max()-sig.min())
        T[i,:] = sig_norm
    
    for i in range(n_points):
        for j in range(i, n_points):
            if i != j:
                correlations[i, j] = np.corrcoef(T[i], T[j])[0, 1]
                correlations[j, i] = correlations[i, j]  # Fill the symmetric entry
                z_avg[j,i] = np.mean([points[i,0],points[j,0]])
                z_avg[i,j] = z_avg[j,i]
                xy_dist[j,i] = np.sqrt(((points[i,1]-points[j,1])**2) + ((points[i,2]-points[j,2])**2))
                xy_dist[i,j] = xy_dist[j,i]
 
    # Extract the upper triangle (excluding the diagonal) of both matrices
    upper_tri_indices = np.triu_indices(n_points, k=1)
    distances_upper = distances[upper_tri_indices]
    correlations_upper = correlations[upper_tri_indices]
    z_avg_upper = z_avg[upper_tri_indices]
    xy_dist_upper = xy_dist[upper_tri_indices]

    # Plot the correlation coefficients as a function of the distances
    plt.figure(figsize=(10, 6))
    plt.scatter(distances_upper, correlations_upper, alpha=0.6)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.grid(True)
    plt.show()

    return (distances_upper, correlations_upper, z_avg_upper, xy_dist_upper)


def extract_time_series_from_vols( volume_time_series, roi_labels):
    """
    Calculate the mean time series for each ROI and return as a 2D array.

    Parameters:
    volume_time_series (np.ndarray): 4D array with shape (time, x, y, z).
    roi_labels (np.ndarray): 3D array with shape (x, y, z) where each voxel is labeled with an ROI.

    Returns:
    np.ndarray: A 2D array where rows are ROIs and columns are time points.
    """
    unique_labels = np.unique(roi_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude the background label (assumed to be 0)

    num_rois = len(unique_labels)
    num_time_points = volume_time_series.shape[0]
    
    # Initialize an array to store the mean time series for each ROI
    roi_time_series = np.zeros((num_rois, num_time_points))
    
    for idx, label in enumerate(unique_labels):
        # Get the mask for the current ROI
        roi_mask = roi_labels == label
        
        # Extract the time series for all voxels in this ROI
        roi_voxels = volume_time_series[:, roi_mask]

        # Calculate the mean time series for this ROI
        mean_time_series = np.mean(roi_voxels, axis=1)
        
        # Store the mean time series in the output array
        roi_time_series[idx, :] = mean_time_series

    return roi_time_series

def calculate_bins_and_means( distance, correlation, bin_width = 10):
    min_distance = np.min(distance)
    max_distance = np.max(distance)
    bin_edges = np.arange(min_distance, max_distance + bin_width, bin_width)
    
    # Step 2: Digitize the distance values
    bin_indices = np.digitize(distance, bins=bin_edges, right=True)
    
    # Step 3: Calculate the mean correlation for each bin
    bin_means = []
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_means.append(correlation[bin_mask].mean())
        else:
            bin_means.append(np.nan)  # Use np.nan for empty bins
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert bin_means to a numpy array for easier handling
    bin_means = np.array(bin_means)

    return (bin_centers, bin_means)


def calculate_pixelwise_snr( LForVol_time_series, ufilt_size = 6, cast_unit8=True ):
    S = LForVol_time_series
    
    # Compute uniform filter
    Sfilt = uniform_filter(S, size=ufilt_size, axes=(0) )
    
    # Compute min
    Smin = Sfilt.min(axis=0)
    
    # Compute df
    df = Sfilt - Smin
    
    # high pass filter
    Shpf = S - Sfilt
    
    # Compute df_max
    df_max = np.max(df, axis = 0 )
    
    # Compute noise, N
    N = np.std( Shpf, axis = 0 )
    
    # Compute SNR
    snr = df_max / N
    
    if cast_unit8:
        # Normalize and cast to uint8
        snr_uint8 = (snr / snr.max() * 255.).round().astype(np.uint8)
        return snr_uint8
    else:
        return snr

def plot_timeseries_heatmap(sigs_dff, fr = 100., norm_max = True ):
    time    = np.arange( np.shape(sigs_dff)[0] )/fr
    cells   = np.arange( np.shape(sigs_dff)[-1] )
    
    fig, ax = plt.subplots()
    
    if (norm_max == True):
        im = ax.pcolormesh( time, cells, (sigs_dff / np.max(sigs_dff, axis = 0)).transpose(), cmap='Blues' )
    else:
        im = ax.pcolormesh( time, cells, sigs_dff.transpose(), cmap='Blues' )
    
    cbar = ax.figure.colorbar(im, ax = ax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel( 'Time (s)' )
    ax.set_ylabel( 'Cell #')

    plt.savefig(r'/Users/starz/Documents/data_analysis/jGCaMP8f/e211130s1a2d1/e211130s1a2d1_heatMapNorm', format='eps')
    
    return (fig, ax)

def segment_vol( vol, spot_sigma_input=3, outline_sigma_input=1):
    device = cle.select_device("Intel(R)")
    recon_vol = np.squeeze( vol )
    [A,B,C] = recon_vol.shape

    voxel_size_x = 1
    voxel_size_y = 1
    voxel_size_z = 1

    input_gpu = cle.push(recon_vol)

    resampled = cle.create([int(input_gpu.shape[0] * voxel_size_z), int(input_gpu.shape[1] * voxel_size_y), int(input_gpu.shape[2] * voxel_size_x)])

    cle.scale(input_gpu, resampled, factor_x=voxel_size_x, factor_y=voxel_size_y, factor_z=voxel_size_z, centered=False)

    no_bg = resampled

    segmented = cle.voronoi_otsu_labeling(no_bg, spot_sigma = spot_sigma_input, outline_sigma = outline_sigma_input)

    return segmented

def segment_vol_otsu( vol ):
    # Example 3D array (replace with your actual data)
    input_array = vol
    
    # Flatten the 3D array to 1D and compute the Otsu threshold
    flat_array = input_array.flatten()
    otsu_threshold = threshold_otsu(flat_array)
    
    # Apply the Otsu threshold to create a binary volume
    binary_volume = np.where(input_array >= otsu_threshold, 1, 0)
    
    # Segment the binary volume using connected component labeling
    labeled_volume, num_features = label(binary_volume)

    return labeled_volume

def segment_vol_binary_threshold( vol, binary_threshold, numLayersZero = 1, min_size = 5 ):
    # Step 2: Define your input array
    input_array = vol
    
    # Step 3: Define the threshold value
    threshold = binary_threshold
    
    # Step 4: Use numpy.where to create the new array
    output_array = np.where(input_array >= threshold, 1, 0)

    # Segment the binary volume using connected component labeling
    labeled_volume, num_features = label(output_array)

    if( numLayersZero > 0 ):
        # Set the outermost numLayersZero layers to zero
        # labeled_volume[:numLayersZero, :, :] = 0         # Front layers
        # labeled_volume[-numLayersZero:, :, :] = 0        # Back layers
        labeled_volume[:, :numLayersZero, :] = 0         # Top layers
        labeled_volume[:, -numLayersZero:, :] = 0        # Bottom layers
        labeled_volume[:, :, :numLayersZero] = 0         # Left layers
        labeled_volume[:, :, -numLayersZero:] = 0        # Right layers

    # Filter out small regions
    for region_label in range(1, num_features + 1):
        region_size = np.sum(labeled_volume == region_label)
        if region_size < min_size:
            labeled_volume[labeled_volume == region_label] = 0
    
    # Re-label the volume to have consecutive labels after filtering
    filtered_labeled_volume, new_num_features = label(labeled_volume > 0)

    return filtered_labeled_volume


def extract_timeSeries( segmented, vol, LFvideo ):

    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)

    num_neurons = cle.pull(segmented).max()

    [B,C,D] = cle.pull(segmented).shape
    A = num_neurons


    total_vol = np.zeros((A,C,D,B))

    tvol = np.zeros([num_neurons, 53, 321, 321])
    for i in np.arange(1,num_neurons+1):
        tvol[i-1,:,:,:] = cle.pull(segmented == i)*150

    total_vol = np.transpose( tvol, [0,2,3,1])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Parametersß
    s=3
    N_=19
    nDepths=53
    V=51
    nLayers=6
    centDpt=nDepths//2
    L2=17
    haarL=8
    nLFs=28
    l=3
    c=400

    frames = LFvideo.shape[0]

    # Set this to path and filename to LFM forward model CNN weights 
    weights_fileFl = r'/Volumes/home/jGCaMP8f_analysis/jGCaMP8f_paperData/weights/epochFl_24.pth'

    Fl=multConvFModel(nDepths=nDepths,s=s,V=V,NxN=N_*N_,haarL=haarL,l=l,c=c).to(device)

    state_dict=Fl.state_dict()

    for n, p in torch.load(weights_fileFl, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)

    lfTrainFTmp = total_vol.astype(np.uint8)

    LFfootprints = np.zeros((A,361,107,107))

    for i in range(0,A):
        print('Neuron ', i)
        
        lfTrain = total_vol[i,:,:,:]
        lfTrain=torch.from_numpy(lfTrain).to(device)
        
        lfTmp_=torch.zeros((361,107,107),dtype=torch.uint8).to(device)# uint8 to save memory
        maxVal=torch.zeros((1,1),dtype=torch.uint8).to(device) # initialize maxVal to 0
        
        lfTmp = lfTrain[:,:,:].float()
        lfTmp = torch.permute(lfTmp,(2,0,1))
        volTmp = Fl(lfTmp[None,:,:,:])
        tmpMax = volTmp.max()
        if tmpMax>maxVal:
            maxVal = tmpMax
        
        # Getting LF
        lfTmp = lfTrain[:,:,:].float()
        lfTmp = torch.permute(lfTmp,(2,0,1))
        volTmp = torch.nn.functional.relu(Fl(torch.nn.functional.pad(lfTmp[None,:,:,:],((L2//2)*s,(L2//2)*s,(L2//2)*s,(L2//2)*s),'reflect')))
        volTmp[volTmp>maxVal] = maxVal # Capping range of values?
        volTmp = 255*volTmp/maxVal #normalization to match uint8
        lfTmp_ = volTmp.detach().cpu().numpy()
        lfTmp_ = np.squeeze(lfTmp_)
        LFfootprints[i,:,:,:] = lfTmp_

    
    D = 19
    E = 107
    
    LFout = np.reshape(LFvideo,(frames,D,D,E,E),order='F')
    LFout = np.transpose(LFout,(0,1,3,2,4))
    LFout = np.reshape(LFout,(frames,2033,2033),order='F')
    LFout = np.transpose(LFout,(1,2,0))
    
    A,B,C = LFout.shape
    LFout = np.reshape(LFout,(A*B,C),order='F')

    N = np.array([])
    neurs = LFfootprints
    X = len(neurs)

    for i in range(0,X):
        neur = neurs[i]
    
        neur = np.reshape(neur,(D,D,E,E),order='F')
        neur = np.transpose(neur,(0,2,1,3))
        neur = np.reshape(neur,(D*D*E*E,1),order='F')
    
        N = np.hstack([N, neur]) if N.size else neur

    N = np.where(np.isfinite(N), N, 0)

    N_pinv = np.linalg.pinv(N)

    T = np.matmul(N_pinv,LFout)

    return (T,N)

# for matching SID and Lnet timeseries
def max_pcc_indices_and_values(array1, array2):
    # Number of rows in each array
    num_ts1 = array1.shape[0]
    num_ts2 = array2.shape[0]

    # Initialize arrays to store the results
    max_indices = np.zeros(num_ts1, dtype=int)
    max_pcc_values = np.zeros(num_ts1)

    # Loop through each time series in array1
    for i in range(num_ts1):
        # Compute PCC between the i-th time series of array1 and all time series in array2
        pcc_values = [np.corrcoef(array1[i], array2[j])[0, 1] for j in range(num_ts2)]
        
        # Find the index of the maximum PCC value
        max_index = np.argmax(pcc_values)
        
        # Store the index and the corresponding max PCC value
        max_indices[i] = max_index
        max_pcc_values[i] = pcc_values[max_index]

    return max_indices, max_pcc_values

def test_spatial_proximity(max_pcc_indices, coords1, coords2, threshold=10):
    """
    Test whether the cells corresponding to the maximum PCC values are within a threshold distance.
    
    Parameters:
    - max_pcc_indices: indices of time series with max PCC (output of max_pcc_indices function)
    - coords1: 2D array of spatial coordinates (n, 3) for the time series in the first array (x, y, z)
    - coords2: 2D array of spatial coordinates (n, 3) for the time series in the second array (x, y, z)
    - threshold: the distance threshold for comparison (default is 10 units)
    
    Returns:
    - A tuple containing:
      - A boolean array indicating whether the distance is less than the threshold for each pair.
      - An array of distances for each pair.
    """
    assert coords1.shape[1] == 3 and coords2.shape[1] == 3, "Coordinates must be 3D (x, y, z)"
    
    proximity_results = []
    distances = []
    
    for i, max_idx in enumerate(max_pcc_indices):
        coord1 = coords1[i]
        coord2 = coords2[max_idx]
        
        # Compute the Euclidean distance between the two points
        distance = np.linalg.norm(coord1 - coord2)
        
        # Store the distance and check if it is less than the threshold
        distances.append(distance)
        proximity_results.append(distance < threshold)
    
    return np.array(proximity_results), np.array(distances)

# Function to normalize a time series between 0 and 1
def normalize(ts):
    return (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

# Function to plot time series pairs with max PCC and their indices
def plot_max_euclidean_pairs(array1, array2, max_indices, distances, pcc_values):
    num_ts1 = array1.shape[0]
    
    plt.figure(figsize=(10, num_ts1 * 4))  # Adjust figure size for multiple subplots
    
    for i in range(num_ts1):
        ts1 = normalize(array1[i])
        ts2 = normalize(array2[max_indices[i]])
        
        # Create a subplot for each pair
        plt.subplot(num_ts1, 1, i + 1)
        
        # Plot both time series
        plt.plot(ts1, label=f'Time Series {i} from Array1', color='blue', linewidth=2)
        plt.plot(ts2, label=f'Time Series {max_indices[i]} from Array2', color='orange', linewidth=2)
        
        # Set plot titles and labels, including distance and PCC
        plt.title(f"Time Series {i} from Array1 and Time Series {max_indices[i]} from Array2\n"
                  f"Distance: {distances[i]:.2f}, PCC: {pcc_values[i]:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Normalized Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

def filter_and_order_time_series(ts_arraySID, centroids_SID, ts_LnetMatrix, centroidsLnet, ts_LnetROI, ts_RL, ts_RL8, pcc_values, distances, max_pcc_indices, threshold_pcc=0.5, threshold_distance=20):
    # Sort both time series arrays based on the max_pcc_indices
    sorted_indices = max_pcc_indices
    
    sorted_ts_array1 = ts_arraySID
    sorted_centroids_array1 = centroids_SID
    sorted_ts_LnetMatrix = np.zeros( sorted_ts_array1.shape)
    sorted_centroids2 = np.zeros( sorted_centroids_array1.shape)
    sorted_ts_LnetROI = np.zeros( sorted_ts_array1.shape )
    sorted_ts_RL = np.zeros( sorted_ts_array1.shape )
    sorted_ts_RL8 = np.zeros( sorted_ts_array1.shape )
    for n in range(sorted_ts_array1.shape[0]):
        sorted_ts_LnetMatrix[n,:] = ts_LnetMatrix[sorted_indices[n], :]
        sorted_centroids2[n,:] = centroidsLnet[sorted_indices[n], :]
        sorted_ts_LnetROI[n,:] = ts_LnetROI[sorted_indices[n], :]
        sorted_ts_RL[n,:] = ts_RL[sorted_indices[n], :]
        sorted_ts_RL8[n,:] = ts_RL8[sorted_indices[n], :]
        
    sorted_pcc_values = pcc_values
    sorted_distances = distances

    # Find indices where PCC > 0.5 and distance < 20
    valid_indices = np.where((sorted_pcc_values > threshold_pcc) & (sorted_distances < threshold_distance))[0]

    # Debugging: print valid indices and their counts
    print("Valid Indices (PCC > 0.5 and Distance < 20):", valid_indices)
    print("Number of valid indices:", valid_indices.size)

    # Check if there are valid indices
    if valid_indices.size == 0:
        print("No valid time series found. Returning empty arrays.")
        return np.empty((0, ts_array1.shape[1])), np.empty((0, ts_array2.shape[1]))

    # Filter sorted time series arrays based on valid indices
    filtered_ts_arraySID = sorted_ts_array1[valid_indices]
    filtered_centroids_arraySID = sorted_centroids_array1[valid_indices]
    filtered_ts_LnetMatrix = sorted_ts_LnetMatrix[valid_indices]
    filtered_centroids_LnetROI = sorted_centroids2[valid_indices]
    filtered_ts_LnetROI = sorted_ts_LnetROI[valid_indices]
    filtered_ts_RL = sorted_ts_RL[valid_indices]
    filtered_ts_RL8 = sorted_ts_RL8[valid_indices]

    return filtered_ts_arraySID, filtered_centroids_arraySID, filtered_ts_LnetMatrix, filtered_centroids_LnetROI, filtered_ts_LnetROI, filtered_ts_RL, filtered_ts_RL8

def plot_timeSeries( timeSeries, napariLabelLayer, frate = 100., filter_sigma = 2, fname = None ):
    T = timeSeries
    fig, ax = plt.subplots(figsize=(6,9))
    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('Cell Number', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    for i in range(T.shape[0]):
        if( filter_sigma > 0):
            sig = ndi.gaussian_filter1d(T[i,:],sigma = filter_sigma, axis = 0) 
        else:
            sig = T[i,:]
        ax.plot(np.arange(0,T.shape[1]/frate,1/frate),((sig-sig.min())/(sig.max()-sig.min()))*0.9+i, color = napariLabelLayer.get_color(i+1))

    if( fname != None ):
        plt.savefig( fname )
    
    fig.show()

    return (fig, ax)

def plot_timeSeries_with_clusters(timeSeries, ordered_clusters, order, frate=100., filter_sigma=2, fname=None):
    """
    Plot time series reordered according to the provided clusters with the same colors as the clusters.
    
    Parameters:
    timeSeries (numpy array): Array of shape (n_neurons, n_time_points).
    napariLabelLayer: Layer that contains color information for each neuron.
    clusters (numpy array): Array of shape (n_neurons,) containing the cluster assignments for each neuron.
    frate (float): Frame rate of the recording.
    filter_sigma (float): Sigma value for Gaussian filter to smooth the time series.
    fname (str): Filename to save the figure (optional).
    """
    
    T = timeSeries

    # Reorder the time series based on the hierarchical clustering order
    T_ordered = T[order, :]
#    ordered_clusters = ordered_clusters[order]

    # Get a color palette for the number of unique clusters
    unique_clusters = np.unique(ordered_clusters)
    colors = sns.color_palette("tab10", n_colors=len(unique_clusters))

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('Cell Number', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)

    # Plot each neuron's time series in the order with the corresponding color
    for i, idx in enumerate(order):
        sig = ndi.gaussian_filter1d(T_ordered[i, :], sigma=filter_sigma, axis=0)
        color = colors[ordered_clusters[i]-1]  # Color based on cluster
        ax.plot(np.arange(0, T.shape[1] / frate, 1 / frate), 
                ((sig - sig.min()) / (sig.max() - sig.min())) * 0.9 + i, 
                color=color)
        
    if fname is not None:
        plt.savefig(fname)
    
    plt.show()

    return fig, ax


def plot_timeseries_heatmap_with_clusters(sigs_dff, clusters, fr=100., norm_max=True):
    """
    Plot a heatmap of the time series with each row (cell) colored differently based on cluster assignments.
    
    Parameters:
    sigs_dff (numpy array): Array of shape (n_time_points, n_neurons) containing the dF/F time series data.
    clusters (numpy array): Array of shape (n_neurons,) containing cluster assignments for the neurons.
    fr (float): Frame rate of the recording.
    norm_max (bool): Whether to normalize the data across neurons by their maximum value.
    """
    
    # Define time and cell ranges for plotting
    time = np.arange(np.shape(sigs_dff)[0]) / fr
    cells = np.arange(np.shape(sigs_dff)[-1])

    fig, ax = plt.subplots()

    # Normalize the signals across neurons, if specified
    if norm_max:
        sigs_dff_normalized = sigs_dff / np.max(sigs_dff, axis=0)
    else:
        sigs_dff_normalized = sigs_dff

    # Plot the base heatmap with intensity variations (Blues cmap for signal intensity)
    im = ax.pcolormesh(time, cells, sigs_dff_normalized.transpose(), cmap='Greys', shading='auto')

    # Get a unique color for each cluster and make it transparent
    unique_clusters = np.unique(clusters)
    colors = sns.color_palette("tab10", n_colors=len(unique_clusters))  # Color for each cluster

    # Overlay transparent color on each row based on its cluster
    for i in range(sigs_dff_normalized.shape[1]):  # Iterate over neurons (rows)
        cluster_idx = np.where(unique_clusters == clusters[i])[0][0]
        color = colors[cluster_idx]
        rgba_color = (*color, 0.3)  # Add alpha transparency to the cluster color

        # Create a transparent colored rectangle for each row (full time range)
        ax.add_patch(plt.Rectangle((time[0], i), time[-1] - time[0], 1, color=rgba_color, lw=0))

    # Remove unwanted spines and set labels
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')

    plt.show()
    return fig, ax
    

def plot_timeSeriesArray( timeSeriesArray, frate = 100., filter_sigma = 2 ):
    fig, ax = plt.subplots(figsize=(8,42))
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cell Number')
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    colormap = plt.cm.plasma
    colors = [colormap(i) for i in np.linspace(0, 1, len(timeSeriesArray))]
    for n in range(len(timeSeriesArray)):
        T = timeSeriesArray[n]
        for i in range(T.shape[0]):
            #sig = ndi.gaussian_filter1d(T[i,:],sigma = filter_sigma, axis = 0) 
            sig = T[i,:]
            ax.plot(np.arange(0,T.shape[1]/frate,1/frate),((sig-sig.min())/(sig.max()-sig.min()))+i, color = colors[n])
    fig.show()

    return (fig, ax)

def plot_tracesFrom2DnumpyArray( T ):
    for i in range(T.shape[0]):
        sig = ndi.gaussian_filter1d(T[i,:],sigma = 1, axis = 0) 
        plt.plot(((sig-sig.min())/(sig.max()-sig.min()))+i)
    plt.show()

def make_SIDlabels( centroids ):
    points = centroids
    # Define the shape of the label image
    label_shape = (53, 321, 321)
    labels = np.zeros(label_shape, dtype=int)
    
    # Round the coordinates to the nearest integer and assign unique labels to each point
    for idx, (z, x, y) in enumerate(points, start=1):
        z, x, y = int(round(z)), int(round(x)), int(round(y))
        if 0 <= z < label_shape[0] and 0 <= x < label_shape[1] and 0 <= y < label_shape[2]:
            labels[z, x, y] = idx
    
    dilation_iterations = 3  # Number of dilation iterations to increase point size
    for idx in range(1, len(points) + 1):
        labels = binary_dilation(labels == idx, iterations=dilation_iterations) * idx + labels * (labels != idx)

    return labels

def show_volAndlabels( vol, labels, namestr ):
    viewer = napari.Viewer()
    viewer.theme = 'dark'
    viewer.dims.ndisplay = 3
    viewer.add_image(vol, name = namestr, scale=(2, 1.667, 1.667) )
    layer = viewer.add_labels(labels.astype("uint8"), name = namestr, scale=(2, 1.667, 1.667))
    layer.bounding_box.visible = True
    layer.bounding_box.points = False
    layer.bounding_box.line_color = 'white'
    layer.bounding_box.line_thickness = 2

    return (viewer, layer)

    
    

    
    