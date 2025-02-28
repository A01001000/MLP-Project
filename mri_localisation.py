import os
import random
import nibabel
import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

class MRILocaliser:
    # Axial (top to bottom) corresponds to dimension 0
    # Coronal (front to back) corresponds to dimension 1
    # Sagittal (left to right) corresponds to dimension 2

    ##  Main methods
    def __init__(self, data_path, target_shape=(20, 100, 100)):
        self.CROP_SENSITIVITY = 10
        self.TARGET_SHAPE = target_shape

        self.data_path = data_path
        self.data_array = None
        self.augmented_data_array = None
        self.basic_mask = None
        self.convex_mask = None

        self.data = nibabel.load(data_path)
        self.data_array = self.data.get_fdata()
        self.cropped_data_array = self.data_array.copy()
        self.augmented_data_array = self.data_array.copy()
    
    def localise(self):
        ''' Controller for the localisation algorithm.'''
        self._threshold_and_rescale_data()
        self._crop_to_target_shape()
        self._crop_to_hippocampus_guess()

    def _crop_to_target_shape(self):
        ''' These functions won't work independently or in different orders.'''
        self._remove_slices_with_shoulders()
        self._crop_coronal()
        self._crop_sagittal()
        self._crop_axial()

    def save_original_images(self, output_dir):
        self.save_array(self.data_array)
    def save_cropped_images(self, output_dir):
        self.save_array(self.cropped_data_array)
        
    def save_array(self, array):
        type = self.data_path.split("/")[-2]
        print(type)
    ## Localisation methods
    def _crop_coronal(self):
        ''' Crops the outer coronal slices by considering their pixel intensities.
            More specifically, sufficiently bright slices are recorded,
            then the image is cropped to contain the outer bright slices.
            
            Uses crop_senstivity; more sensitive means more slices are included.'''
        intense_slices = []
        crop_threshold = self.augmented_data_array.shape[0] * self.augmented_data_array.shape[1] / self.CROP_SENSITIVITY
        for i in range(self.augmented_data_array.shape[1]):
            if np.sum(self.augmented_data_array[:,i,:]) > crop_threshold:
                intense_slices.append(i)
        i1 = intense_slices[0]; i2 = intense_slices[-1]
        size_diff = self.TARGET_SHAPE[1] - (i2 - i1)
        i1 -= size_diff // 2
        i2 = i1 + self.TARGET_SHAPE[1]
        self._crop_arrays(coronal_indices=(i1, i2))    

    def _crop_sagittal(self):
        ''' Crops the outer sagittal slices by considering their pixel intensities.
            More specifically, sufficiently bright slices are recorded,
            then the image is cropped to contain the outer bright slices.
            
            Uses crop_senstivity; more sensitive means more slices are included.'''
        intense_slices = []
        crop_threshold = self.augmented_data_array.shape[0] * self.augmented_data_array.shape[1] / self.CROP_SENSITIVITY
        for i in range(self.augmented_data_array.shape[2]):
            if np.sum(self.augmented_data_array[:,:,i]) > crop_threshold:
                intense_slices.append(i)
        i1 = intense_slices[0]; i2 = intense_slices[-1]
        size_diff = self.TARGET_SHAPE[2] - (i2 - i1)
        i1 -= size_diff // 2
        i2 = i1 + self.TARGET_SHAPE[2]
        self._crop_arrays(sagittal_indices=(i1, i2))

    def _crop_axial(self):
        ''' Heuristically crops outer axial slices based on the slice's corresponding convex mask.
            Slices are included based on a threshold of 3/4, or the maximal end slice intensity.
            The latter condition helps keep the mask centred if someone has high shoulders.
            or if they're wearing a hat?'''
        self.convex_mask = self._create_convex_mask()
        convex_sums = self._calculate_convex_slice_sums()
        highest_end = max(np.min(convex_sums[:convex_sums.shape[0] // 2]), 
                          np.min(convex_sums[convex_sums.shape[0] // 2:]))
        intense_slices = np.where(convex_sums > max(np.max(convex_sums) * 3/4, highest_end))[0]
        slice_num = self.TARGET_SHAPE[0] * 3 # How many slices to have left after cropping
        i1 = intense_slices[0]; i2 = intense_slices[-1]
        size_diff = slice_num - (i2 - i1)
        i1 -= size_diff // 2
        i2 = i1 + slice_num
        self._crop_arrays(axial_indices=(i1, i2))
    
    def _guess_central_hippocampus_slice(self):
        ''' Locates the axial centre of the hippocampus, based on the sagittal brightness profile.
            More specifically, the hippocampus slices are bright at the edges and have high value.
            10-20 slides slices after the hippocampus, the edges become dark.
            The algorithm looks for this transition to locate the hippocampus.
            '''
        
        slice_brightnesses = self._calculate_sagittal_weighted_slice_brightnesses()
        likely_gap = (10, 20) # Shouldn't really be hard-coded, instead should depend on head height
        diffs = np.zeros(len(slice_brightnesses))
        for i in range(0, len(slice_brightnesses)):
            greatest_diff = 0
            for j in range(likely_gap[0], likely_gap[1]):
                if i + j < len(slice_brightnesses):
                    diff = slice_brightnesses[i] - slice_brightnesses[i + j]
                    if diff > greatest_diff:
                        greatest_diff = diff
            diffs[i] =(greatest_diff)
        guess = np.argmax(diffs) - 5 # Small adjustment to find the hippocampus centre
        return round(guess)

    def _crop_to_hippocampus_guess(self):

        guess = self._guess_central_hippocampus_slice()
        lower = guess - self.TARGET_SHAPE[0] // 2
        upper = lower + self.TARGET_SHAPE[0]
        self._crop_arrays(axial_indices=(lower, upper))

    def _crop_arrays(self, axial_indices=(-1, np.inf), coronal_indices=(-1, np.inf), sagittal_indices=(-1, np.inf)):
        ''''Safely change array dimensions'''
        aug_shape = self.augmented_data_array.shape
        ai = (int(max(0, axial_indices[0])), int(min(axial_indices[1], aug_shape[0])))
        ci = (int(max(0, coronal_indices[0])), int(min(coronal_indices[1], aug_shape[1])))
        si = (int(max(0, sagittal_indices[0])), int(min(sagittal_indices[1], aug_shape[2])))

        self.augmented_data_array = self.augmented_data_array[ai[0]:ai[1], ci[0]:ci[1], si[0]:si[1]]
        self.data_array = self.data_array[ai[0]:ai[1], :, :]
        self.cropped_data_array = self.cropped_data_array[ai[0]:ai[1], ci[0]:ci[1], si[0]:si[1]]

    ## Helper methods        
    def _calculate_sagittal_weighted_slice_brightnesses(self):
        ''' Calculate the saggital-weighted brightness of each axial slice.
            More precisely, the weight is based on distance from the slice's central "column".
            The values are then smoothed to reduce noise.'''
        
        brightnesses = np.zeros(self.augmented_data_array.shape[0])
        centre = self.augmented_data_array.shape[2] / 2.0
        for i in range(self.augmented_data_array.shape[0]):
            brightness = 0
            for j in range(self.augmented_data_array.shape[2]):
                row_weight = abs(j - centre)
                brightness += np.sum(self.augmented_data_array[i, :, j] * row_weight)
            brightnesses[i] = brightness
        smooth_width = 5
        brightnesses_padded = np.pad(brightnesses, pad_width=smooth_width, mode='edge')
        brightnesses_smooth = np.convolve(brightnesses_padded, np.ones(smooth_width*2)/(smooth_width*2), mode='same')
        brightnesses_smooth = brightnesses_smooth[smooth_width:-smooth_width]
        return brightnesses_smooth/np.max(brightnesses_smooth)
    
    def _calculate_convex_slice_sums(self):
        ''' Calculate the intensity sum of each axial slice's convex mask.
            The values are then smoothed to reduce noise.'''
        sums = [np.sum(self.convex_mask[i,:,:]) for i in range(self.convex_mask.shape[0])]
        sums = np.convolve(sums, np.ones(10)/10, mode='valid')
        return sums
    
    ## Image processing
    def _create_basic_mask(self):
        return np.where(self.augmented_data_array > 0, 1, 0)
    
    def _create_convex_mask(self):
        ''' For every axial slice, construct the minimal convex mask (convex shape containing all points).
            This produces a mask with no holes to estimate the head's cross-section.'''
        mask = self.augmented_data_array
        convex_mask = np.zeros(mask.shape)
    
        for i in range(convex_mask.shape[0]):
            slice = mask[i, :, :].copy()
            slice = (slice * 255).astype(np.uint8)
            slice = cv2.morphologyEx(slice, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
            
            # I'm not sure why I need the contours to calculate the hull, but other methods seem to fail
            contours, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.vstack(contours)
                hull = cv2.convexHull(all_points)
                cv2.drawContours(slice, [hull], -1, (1), thickness=cv2.FILLED)
            convex_mask[i, :, :] = slice

        return convex_mask
    
    def _threshold_and_rescale_data(self):
        ''' Sets up the augmented_data_array, which is always used in place of the original data_array.
            Noise is first removed by thresholding at double the data's mean (which is typically low due to black background).
            It seems logical to use standard deviations, but it didn't work as well. I don't remember why.
            A new mean (without black pixels) is calculated to threshold abnormally high values.
            Finally, a binary mask is constructed.
            By masking to binary pixel values, extreme values are mitigated so the algorithm is more robust to noise.'''
        self.augmented_data_array = self.augmented_data_array/np.mean(self.augmented_data_array)
        thresholded_array = np.where(self.augmented_data_array > 2, self.augmented_data_array, np.nan)
        mean = np.nanmean(thresholded_array)
        self.augmented_data_array = np.where(thresholded_array > 0, self.augmented_data_array, 0)
        self.augmented_data_array = np.where(abs(self.augmented_data_array - mean) < 1.5 * mean, self.augmented_data_array, 0)
        self.augmented_data_array = self._create_basic_mask()

    def _remove_slices_with_shoulders(self):
        ''' Shoulders can really confuse the algorithm (slicing will instead crop to shoulders, mean pixel value is skewed).
            This function removes the shoulders by calculating axial slice "centres of mass" (maybe not a good description?).
            Where the centre of mass is far from the middle of the slice, the slice is cropped to remove the likely shoulders.'''
        valid_indices = []
        valid_centres = []
        for i in range(self.augmented_data_array.shape[0]):
            slice = self.augmented_data_array[i,:,:]
            row_weights_summed = 0

            slice_sum = np.sum(slice)
            if slice_sum > 0:
                for j in range(self.augmented_data_array.shape[1]):
                    row_sum = np.sum(slice[j,:])
                    row_weights_summed += j * row_sum
                row_weights_summed /= slice_sum
                valid_indices.append(i)
                valid_centres.append(row_weights_summed)

        valid_centres = np.convolve(valid_centres, np.ones(10)/10, mode='valid')
        centres_mean = np.mean(valid_centres)
        centred_slices = []
        for i in range(valid_centres.shape[0]):
            # 20 pixels probably shouldn't be hardcoded
            if abs(valid_centres[i] - centres_mean) < 20:
                centred_slices.append(valid_indices[i])
        self._crop_arrays(axial_indices=(centred_slices[0], centred_slices[-1]-5))

        return valid_centres
