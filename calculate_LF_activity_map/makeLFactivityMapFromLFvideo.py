import numpy as np
import matplotlib.pyplot as plt
import tifffile
import jGCaMP8f_helperFuns as hpf



# Calculates the LF activity map and mean LF (across time) from an LF video.  The mean LF is to make rectifying the LFs easier in the next step.
def calc8_mean_activeCellsSat1024( fname ):
    lfts = tifffile.imread( fname )
    active_cells_LF = hpf.calc_active_cells_LF( lfts )
    active_cells_LF_sat1024 = hpf.rescale_array(active_cells_LF, satVal=1024)
    Smean = np.mean(lfts,axis=0)
    Smean8 = hpf.scale_cast_uint8(Smean)
    plt.imshow(Smean8)
    plt.imshow(active_cells_LF_sat1024)

    return (active_cells_LF_sat1024, Smean8)


# Replace with path and name of LF video
LFvideo_file_path_and_name = r'/Volumes/home/jGCaMP8f_analysis/jGCaMP8f_paperData/211115_s1a2/s1a2d1_LF_1P_1x1_400mA_100Hz_func_500frames_4AP_2/s1a2d1_LF_1P_1x1_400mA_100Hz_func_500frames_4AP_2_MMStack_Default.ome.tif'

(active_cells_LF_sat1024, mean8_LF ) = calc8_mean_activeCellsSat1024( LFvideo_file_path_and_name )

# Replace with paths and filenames for saving the LF activity map and mean LF
output_LFactivityMap_file_path_and_name = r'/Volumes/home/jGCaMP8f_analysis/activeCellsLFs/active_cells_LF_sat1024'
output_meanLF_file_path_and_name = r'/Volumes/home/jGCaMP8f_analysis/activeCellsLFs/mean8_LF'

tifffile.imwrite( output_LFactivityMap_file_path_and_name,  active_cells_LF_sat1024)
tifffile.imwrite( output_meanLF_file_path_and_name, mean8_LF )


