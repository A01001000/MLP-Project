from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mri_localisation import *
import os
import random

# ADNI/AD/I67261.nii
# 

path = "/home/arecibo/mlp-coursework_3/ADNI/CN/I59214.nii"
mri_localiser = MRILocaliser(path)
# mri_localiser._threshold_and_rescale_data()
# mri_localiser._remove_slices_with_shoulders()
mri_localiser.localise()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
plt.subplots_adjust(bottom=0.35)

ax1.set_title(mri_localiser.data_path)
ax2.set_title(mri_localiser.data_path)
ax3.set_title(mri_localiser.data_path)

plot_1 = mri_localiser.data_array
plot_2 = mri_localiser.augmented_data_array
plot_3 = mri_localiser.data_array

if plot_2 is None:
    plot_2 = np.ones(mri_localiser.data_array.shape)
if plot_3 is None:
    plot_3 = np.ones(mri_localiser.data_array.shape)


plot_1 = plot_1/np.max(plot_1)
plot_2 = plot_2/np.max(plot_2)
plot_3 = plot_3/np.max(plot_3)

guess = plot_1.shape[0] // 2

im_1 = ax1.imshow(plot_1[guess,:,:], cmap='gray', vmin=0, vmax=1)
im_2 = ax2.imshow(plot_2[guess,:,:], cmap='gray', vmin=0, vmax=1)
im_3 = ax3.imshow(plot_3[guess,:,:], cmap='gray', vmin=0, vmax=1)

# convex_sums = mri_localiser.calculate_convex_slice_sums()
brightnesses = mri_localiser._calculate_sagittal_weighted_slice_brightnesses()
lower_axes = plt.axes([0.22, 0.05, 0.71, 0.1])
lower_axes.plot(brightnesses, 'r')
# lower_axes.set_title('arbitrary')

ax_slider = plt.axes([0.25, 0.2, 0.35, 0.03])
slider = Slider(ax_slider, 'Frame', 0, plot_1.shape[0] - 1, valinit=guess, valstep=1)

def update(val):
    frame = int(slider.val)
    im_1.set_data(plot_1[frame,:,:])
    im_2.set_data(plot_2[frame,:,:])
    im_3.set_data(plot_3[frame,:,:])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()