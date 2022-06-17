# Image analysis code for micrographs with BCP-like structures (periodic, defined dark and light domains)
# For simplicity, this code does not support images with featureless domains (please see featureless.py if needed)
# Be sure to run the code in the provided .yml environment to avoid version issues

# This code proceeds as follows:

#   1. 	Load images from a folder
#		- use separate code (run_image_analysis.py) to interface with the folder directly
#		- scrape input information from the filenames
#
#   2. 	Find characteristic periodicity of the image 
#		- perform the fast fourier transform (FFT) of the image
#		- perform the radial integration of the FFT and identify the dominant periodicity peak
#		- convert the peak (units of inverse pixels) to a real-space periodicity (units of pixels and nm)
#		- optionally, estimate the periodicity from the polymer's molecular weight, so the correct peak is identified 
#		- PS-b-P4VP values are included in this code; other polymers will need to be added manually
#
#   3. 	Binarize the image
#		- remove small features (dark or light speckles) with a median filter 
#		- remove image-scale intensity variation (ie. a "tilt") by dividing the image by a heavily-blurred version of itself
#		- use Otsu thresholding to create a binary image
#		- tidy the thresholded image by removing small dark or light objects 
#
#   4. Skeletonize the image
#		- reduce the dark and light domains to single-pixel lines which capture the connectivity of the domain structure
#		- tidy the skeletons by removing small branches (skeletonization artifacts)
#
#   5. Sort the components of the skeleton into sets of "dots" and "lines" for dark and light domains
#		- calculated the area fractions of dot and line features
#
#   6. Find the defects in the line components
#		- measure the connectivity of every pixel in the "lines" skeletons
#		- two neighboring pixels = normal, no defect
#		- 3+ neighbors = junction defect
#		- 1 neighbor = end defect
#		- tally the total defect counts to identify the overall image defect densities
#	

# ****************************************************************************************************************
# Dependencies
# ****************************************************************************************************************

import numpy as np
import math
import os
import csv
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Make plot labels extra large by default
params = {'legend.fontsize': 'x-large',
		 'axes.labelsize': 'x-large',
		 'axes.titlesize':'x-large',
		 'xtick.labelsize':'x-large',
		 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import scipy
from scipy import signal
from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from scipy import spatial

import skimage
from skimage import data, color
import skimage.morphology as morphology
import skimage.draw as draw
import skimage.filters
from skimage.filters import threshold_minimum, threshold_otsu, rank, median
from skimage.transform import resize

from PIL import Image 

# Setting up colormaps for visualizations:

def truncate_colormap(cmapIn='CMRmap_r', minval=0.0, maxval=1.0, n=100):   
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap

white_clear = [(1,1,1,1),(0,0,0, 0)]
bin_cmap = colors.LinearSegmentedColormap.from_list(name='custom',colors=white_clear, N=2)
twilight_trun = truncate_colormap(cmapIn='twilight', minval=.15, maxval=.85)

# ****************************************************************************************************************
# Helper Functions
# ****************************************************************************************************************

def running_mean(x, N): #used to smooth out noisy data, source: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N)

def radial_profile(data, center): #used to perform radial integration, source: https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr
	return radialprofile 

def take_closest(num,collection): # returns member of 'collection' nearest to 'num'. source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
	return min(collection,key=lambda x:abs(x-num))

def gauss(x, A, mu, sigma): # Gaussian function to use with model fitting
	return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss_shift(x, A, mu, sigma, shift): # Gaussian function with y-axis shift
		return A*np.exp(-(x-mu)**2/(2*sigma**2))+shift

def neighbors(x,y,image): # Return 8 neighbors of image point P1(x,y), in a clockwise order
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def neighborhood_sum(x,y,image, extent): # sum all pixels around x,y in a square of size 2*extent
	img = image
	sum_total = 0
	for xval in range(x-extent, x+extent): # don't account for values past bounds of the image
		for yval in range(y-extent, y+extent):
			sum_total += image[xval,yval]
	return sum_total

# ****************************************************************************************************************
# Specific Functions
# ****************************************************************************************************************

# Crop Image: remove scalebars and borders around the .jpg AFM images
# This is specific to a Bruker Dimension Edge AFM
def crop_img(img):
	size = img.shape
	if size == (790, 761):
		return img[24:733,6:715]
	elif size == (790, 765):
		return img[24:732,6:716]
	elif size == (759, 734) or size == (759, 730):
		return img[24:702,6:682]
	elif size == (793, 764):
		return img[45:757,5:717]
	elif size == (950, 921) or size == (774, 745) or size == (774, 749) or size == (775, 746) or size == (775, 750) or size == (771, 749):
		return img[24:716,6:698]
	elif size == (811, 901) or size == (811, 905):
		return img[24:716,6:698]
	elif size == (633, 608) or size == (634, 609):
		return img[24:552, 6:551]
	elif size == (633, 604):
		return img[26:575, 7:557]
	elif size == (634, 605):
		return img[26:575, 7:557]
	else:
		print('bad crop')
		return img[45:702,6:682] #if size isn't in this list, avoid a "None" image type

# Characteristic Periodicity: return the characteristic periodicity of the image, in inverse pixels
def char_periodicity(data, width1, height1, ps, pxls_to_um): 
	
	fft_org = np.fft.fftshift(np.fft.fft2(data))
	fft_org = abs(fft_org)

	center_indices = np.unravel_index(fft_org.argmax(), fft_org.shape)

	rad = radial_profile(fft_org, (center_indices[1], center_indices[0]))

	rad_smooth = running_mean(rad, 4)

	ps = int(ps)

	# this is where the periodicity is estimated from the polystyrene molecular weight
	# this step is optional, but useful when the radial integration results in multiple peaks
	if ps == 19:
		target_value = width1*1000/(30*pxls_to_um)
		cutoff_value = 10

	elif ps == 25:
		target_value = width1*1000/(43*pxls_to_um)
		cutoff_value = 12

	elif ps ==47:
		target_value = width1*1000/(55*pxls_to_um)
		cutoff_value = 12

	elif ps == 50:
		target_value = width1*1000/(70*pxls_to_um)
		cutoff_value = 12

	elif ps == 61:
		target_value = width1*1000/(75*pxls_to_um)
		cutoff_value = 20

	elif ps == 75:
		target_value = width1*1000/(80*pxls_to_um)
		cutoff_value = 20

	elif ps == 104:
		target_value = width1*1000/(105*pxls_to_um)
		cutoff_value = 30

	elif ps == 330:
		target_value = width1*1000/(250*pxls_to_um)
		cutoff_value = 30

	elif ps == 650:
		target_value = width1*1000/(600*pxls_to_um)
		cutoff_value = 30

	else:
		print("unknown polymer")
		target_value = width1*1000/(50*pxls_to_um)

	true_max = np.argmax(rad_smooth[(int(target_value/2)):])+ int(target_value/2)

	if np.abs(width1/(true_max+2)*1000/pxls_to_um - width1/target_value*1000/pxls_to_um) < cutoff_value:    # first see if the largest peak comes from the periodicity
		max_value = true_max + 2

	else:                                   # if not, try find_peaks
		max_values = signal.find_peaks_cwt(rad_smooth[1:], np.arange(1,30)) 
		if len(max_values) > 0:
			closest_max = take_closest(target_value-1, max_values) +  1 + 2

			if np.abs(width1/closest_max*1000/pxls_to_um - width1/target_value*1000/pxls_to_um) < cutoff_value:
				max_value = closest_max

			else:
				print('unexpected periodicity: '+str(int(width1*1000/closest_max/pxls_to_um))+' vs. expected '+str(int(width1*1000/target_value/pxls_to_um)))
				max_value = int(target_value)
		else:
			print('unexpected periodicity')
			max_value = int(target_value)	

	# can visualize the peak finding for debugging purposes:
	
	# plt.clf()
	# plt.plot(np.arange(1,len(rad)), rad[1:])
	# plt.plot(np.arange(2,len(rad_smooth)), rad_smooth[1:-1])
	# plt.axvline(max_value, color='lightsteelblue', linestyle='dashed')
	# plt.show()
	# plt.clf()

	# Find the FWHM of the selected peak to perform Scherrer grain size calculation later
	lower_bound = max(max_value-20, 5)

	x, ydata = range(lower_bound, max_value+20), rad[lower_bound:max_value+20]

	try: # occasionally crashes, use 'try' so it won't throw a fatal error
		popt, pcov = scipy.optimize.curve_fit(gauss_shift, x, ydata, p0 =[10000, max_value, 1, 100000])
		FWHM = 2.35482*popt[2] #full width, half max-- from the definition of a Gaussian's FWHM
	except:
		FWHM = -10

	# can visualize the Gaussian fitting for debugging purposes:

	# plt.plot(rad[0:200], color='navy')
	# plt.plot(x, func(x,*popt), 'r-')
	# plt.axvline(max_value, color='lightsteelblue', linestyle='dashed')
	# plt.xlim(lower_bound, max_value+20)
	# plt.ylim(0, 1000)
	# plt.show()
	# plt.clf()

	return max_value, FWHM

# Binarize Data: Flatten intensity variation in the data, and then binarize (dark = P4VP, light = PS)
def binarize_data(data, period):

	smoother = skimage.filters.gaussian(data, period)
	phase_data_flat = data/smoother

	thresh = threshold_otsu(phase_data_flat)
	# thresh = custom_otsu(phase_data_flat)

	return phase_data_flat < thresh

# Tidy Binarized: Remove small features (both dark in light, and light in dark) that are imaging artifacts or nanoparticles
def tidy_bin(bin_image, threshold): #remove items smaller than the threshold

	labels = morphology.label(bin_image, connectivity=1)

	lbls, cts = np.unique(labels, return_counts=True)

	lbls_cts = np.array([lbls[1:], cts[1:]]).T # gives an array of label-count pairs

	good_labels = np.where(cts>threshold, 0, lbls) # set labels belonging to objects with fewer pixels to 0

	good_labels = np.unique(good_labels)

	tidied_bin = np.isin(labels, good_labels)

	# Invert the image and do the same thing for the other phase:

	labels = morphology.label(1-tidied_bin, connectivity=1)

	lbls, cts = np.unique(labels, return_counts=True)

	lbls_cts = np.array([lbls[1:], cts[1:]]).T #gives an array of label-count pairs

	good_labels = np.where(cts>threshold, 0, lbls) #set labels belonging to objects with fewer pixels to 0

	good_labels = np.unique(good_labels)

	tidier_bin = 1-np.isin(labels, good_labels)

	return tidier_bin


def separate_dots(binarized, skeleton, period, block_ratio): # code adapted from https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

	width, height = binarized.shape

	distance = ndi.distance_transform_edt(binarized) # every pixel value comes from its minimum distance from an edge

	skel_dist = distance*skeleton

	# find local minima within skel_dist
	remove_background = np.where(skel_dist != 0, skel_dist, 100)

	mins = morphology.extrema.local_minima(remove_background)

	lowest_mins = np.where(np.logical_and(0 < skel_dist*mins,skel_dist*mins < math.ceil(period*block_ratio*0.4)), 1, 0)

	markers = ndi.label(lowest_mins)[0]

	marks = np.nonzero(markers)
	marks = np.array(marks)
	marksT = np.array(marks).T

	matrix_size = max(1, int(period*block_ratio*0.5))

	for m in marksT:
		if np.sum(skeleton[m[0]-1:m[0]+2, m[1]-1:m[1]+2]) == 3: # make sure it's not the endpoint
			if np.logical_and(m[0]>matrix_size, m[0]< width-matrix_size):
				if np.logical_and(m[1]>matrix_size, m[1]< height-matrix_size): # ignore points on the edge of the image
				
					x = m[0]
					x_min = x-matrix_size
					x_max = x+matrix_size

					y = m[1]
					y_min = y-matrix_size
					y_max = y+matrix_size

					# label the bounding_lines to separate them:
					labelled_bounds, n_labels = morphology.label((1-binarized[x_min:x_max, y_min:y_max]), connectivity=2, return_num=True)

					if n_labels == 2:
						line_1 = np.array(np.where(labelled_bounds==1)).T
						line_2 = np.array(np.where(labelled_bounds==2)).T

						# next, select the pair of points in line_1 and line_2 that are closest together
						dist_matrix = spatial.distance_matrix(line_1, line_2)
						l1, l2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

						# these are the two points!
						pt1x, pt1y = line_1[l1] + np.array([x_min, y_min])
						pt2x, pt2y = line_2[l2] + np.array([x_min, y_min])

						line_seg = draw.line(pt1x, pt1y, pt2x, pt2y)

						# draw the line:
						line_seg = draw.line(pt1x, pt1y, pt2x, pt2y)
						lines_pts = np.array(line_seg).T

						surrounding_pts = [[0,0], [1,0], [1,1], [0,1],[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
						for l_pt in lines_pts:
							for s_pt in surrounding_pts:
								binarized[l_pt[0]+s_pt[0], l_pt[1]+s_pt[1]] = 0


					if n_labels > 2: # use the two largest regions to draw the line
						vals = [0]*n_labels
						for i in range(n_labels):
							vals[i] = np.count_nonzero(labelled_bounds==i)

						sorted_vals = sorted(vals)

						v1 = vals.index(sorted_vals[0]) # index of the first largest region
						v2 = vals.index(sorted_vals[1]) # index of the second largest region

						line_1 = np.array(np.where(labelled_bounds==v1)).T
						line_2 = np.array(np.where(labelled_bounds==v2)).T

						# next, select the pair of points in line_1 and line_2 that are closest together
						dist_matrix = spatial.distance_matrix(line_1, line_2)
						l1, l2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

						# these are the two points!
						pt1x, pt1y = line_1[l1] + np.array([x_min, y_min])
						pt2x, pt2y = line_2[l2] + np.array([x_min, y_min])

						# draw the line:
						line_seg = draw.line(pt1x, pt1y, pt2x, pt2y)
						lines_pts = np.array(line_seg).T

						surrounding_pts = [[0,0], [1,0], [1,1], [0,1],[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
						for l_pt in lines_pts:
							for s_pt in surrounding_pts:
								binarized[l_pt[0]+s_pt[0], l_pt[1]+s_pt[1]] = 0

	return binarized

# Skeletonize: Draw a one-pixel-wide shape through both the dark and light phases, preserving connectivity
def sk_light(data): 
    data = 1 - data # inverts the image
    return morphology.skeletonize(data)

def sk_dark(data): 
    return morphology.skeletonize(data)

# Dots and Lines: Sort elements in the binarized image into 'dots' and 'lines', depending on how large they are
def dots_and_lines(skeleton, full_binary, period):
	skel_pixels = skeleton != 0
	bin_pixels = full_binary != 0

	label_bin = morphology.label(bin_pixels, connectivity=1)
	unique_bin, counts_bin =  np.unique(label_bin, return_counts=True)

	label_skel = np.multiply(skel_pixels, label_bin) # get skeleton filled with labels from the binary image
	unique, counts = np.unique(label_skel, return_counts=True)

	counts_expanded = []

	for i in range(len(counts_bin)):
		if np.isin(unique_bin[i], unique):
			new_val = counts[list(unique).index(unique_bin[i])]
			counts_expanded.append(new_val)
		else:
			counts_expanded.append(0)

	counts = np.array(counts_expanded)

	dot_thresh = period
	
	dot = np.where((counts < dot_thresh), 1, 0) # gives a list where indices correponding to dot labels = 1
	
	dot_labs = np.nonzero(dot)
	dot_image = np.where(np.isin(label_skel, dot_labs), 1, 0) # skeleton only
	all_dots = np.where(np.isin(label_bin, dot_labs), 1, 0)

	label_dots, n_dots = morphology.label(bin_pixels, connectivity=1, return_num=True)

	line = np.insert(np.where(counts[1:] >= dot_thresh, 1, 0),0,0) # adds a zero to make up for removing the first value 
	line_labs = np.nonzero(line)
	line_image = np.where(np.isin(label_skel, line_labs), 1, 0) # skeleton only
	all_lines = np.where(np.isin(label_bin, line_labs), 1, 0)

	return all_dots, all_lines, line_image, n_dots

# Get Skeleton Defects: Find junctions and ends in the skeleton, based on the nearest neighbors of every pixel

# There are two versions of this function
# The first version is quicker and less precise- it counts up neighbors but doesn't account for where they are

def get_skeleton_defects(skeleton): # Given a skeletonized image, it will give the coordinates of the intersections of the skeleton.
	image = skeleton.copy()
	nb_counts = 2*skeleton.copy()

	for x in range(1,len(image)-1):
		for y in range(1,len(image[x])-1): 
			if image[x][y] == 1: # if the point is in the skeleton
				nb_counts[x][y] = sum(neighbors(x, y, image))

	nb_counts = np.array(nb_counts)
	junctions = np.where(nb_counts>=3, 1, 0)
	ends = np.where(nb_counts==1, 1, 0)

	return junctions, ends

# The second version is more precise, used for actually calculating the defect densities. 
# Based on: https://stackoverflow.com/questions/41705405/finding-intersections-of-a-skeletonised-image-in-python-opencv

def get_skeleton_defectsII(skeleton): 
	# A biiiiiig list of valid intersections             2 3 4
	# These are in the format shown to the right         1 C 5
	#                                                    8 7 6 
	validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
						 [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
						 [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
						 [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
						 [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
						 [1,0,0,0,1,0,1,0]];

	validEnd = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
				[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]];
	image = skeleton.copy();

	# make skeleton fat then re-skeletonize to avoid small disjoints
	image = morphology.skeletonize(morphology.dilation(image))

	intersections = np.zeros((image.shape))
	ends = np.zeros((image.shape))

	for x in range(3,len(image)-3):
		for y in range(3,len(image[x])-3):
			# If we have a white pixel
			if image[x][y] == 1:
				nbs = neighbors(x,y,image);
				nb_sum = neighborhood_sum(x, y, intersections, 3)
				if sum(nbs) > 3:
					if nb_sum==0:
						intersections[x,y]=1
				elif nbs in validIntersection:
					if nb_sum==0: # avoid counting intersections that are super close together
						intersections[x,y]=1
				elif nbs in validEnd:
					if neighborhood_sum(x, y, intersections+ends, 3)==0: # avoid counting ends that are super close to ends or intersections
						ends[x,y]=1

	return intersections, ends

def connectivity_results(line_image_light, line_image_dark, width, height):
	junct_light, ends_light = get_skeleton_defects(line_image_light) #list of points of intersection
	junct_dark, ends_dark = get_skeleton_defects(line_image_dark) #list of points of intersection

	return junct_light, junct_dark, ends_light, ends_dark 

def connectivity_resultsII(line_image_light, line_image_dark, width, height):
	junct_light, ends_light = get_skeleton_defectsII(line_image_light) #list of points of intersection
	junct_dark, ends_dark = get_skeleton_defectsII(line_image_dark) #list of points of intersection

	return junct_light, junct_dark, ends_light, ends_dark 

# Prune Skeleton: Remove small skeleton branches, which are mostly artifacts that increase the defect densities

def prune_skel(line_image, insct, threshold): #remove items smaller than the threshold
	
	line_image = line_image.astype(int)-line_image.astype(int)*morphology.binary_dilation(insct)
	label_dd, dd_labels = scipy.ndimage.label(line_image, structure=[[1,1,1],[1,1,1],[1,1,1]])
	
	lbls, cts = np.unique(label_dd, return_counts=True)
	lbls_cts = np.array([lbls[1:], cts[1:]]).T # gives an array of label-count pairs
	
	good_labels = np.where(cts>threshold, 0, lbls) # set labels belonging to objects with fewer pixels to 0
	good_labels = np.unique(good_labels)
	
	pruned_lines = np.isin(label_dd, good_labels)
	pruned_lines = 1-pruned_lines.astype(int)+morphology.binary_dilation(insct)
	pruned_lines = morphology.skeletonize(pruned_lines)

	# need to remove orphaned intersection points

	label_dd, dd_labels = scipy.ndimage.label(pruned_lines, structure=[[1,1,1],[1,1,1],[1,1,1]])
	lbls, cts = np.unique(label_dd, return_counts=True)
	lbls_cts = np.array([lbls[1:], cts[1:]]).T # gives an array of label-count pairs

	good_labels = np.where(cts>threshold, 0, lbls) # set labels belonging to objects with fewer pixels to 0
	good_labels = np.unique(good_labels)

	pruned_lines = 1-np.isin(label_dd, good_labels)

	return pruned_lines

def morphology_map(data, binarized, width, height, period):

	size = int(period*1.0) #size of the sliding window in pixels
	half_size = int(size/2)
	px_size = int(4) #needs to be even 
	half_px = int(px_size/2)

	data_pad = np.lib.pad(data.copy(), ((size, size),(size, size)), 'symmetric')

	morph_map = np.zeros((data_pad.shape))

	for x in np.arange(size, width-size, px_size):
		x = int(x)
		for y in np.arange(size, height-size, px_size):
			y=int(y)
			square = data_pad[(x-size):(x+size),(y-size):(y+size)]
			square_sum = np.sum(square)/(2*size)**2

			morph_value = square_sum*3 + 4
			morph_map[x-half_px:x+half_px,y-half_px:y+half_px] = np.full((px_size, px_size), morph_value)

	morph_map = morph_map[size:width-size, size:height-size]

	new_bin = binarized[size:width-size, size:height-size]

	morph_map = morphology.closing(morphology.closing(morph_map))
	morph_map = morphology.opening(morphology.opening(morph_map))

	return morph_map, new_bin

# ****************************************************************************************************************
# ****************************************************************************************************************
# Final analysis function 
# ****************************************************************************************************************
# ****************************************************************************************************************

# All of these functions are used to process an image:
image_save_location = '/Users/emmavargo/Desktop/'

def analyze_image(image_data, image_name, image_size, save_image=False, show_image=False, save_location=image_save_location):

	# make a folder for saving results	
	if save_image == True:	
		results_folder = save_location+image_name+'/'
		if os.path.exists(results_folder) == False:
			os.mkdir(results_folder) # data will be overwritten if the same image is analyzed multiple times

	# extract sample information from the image name
	image_name = image_name.replace('_','-') # use either a dash or an underscore to separate items
	name_splt = image_name.split('-')
			
	date = name_splt[0]

	if name_splt[1][0] == 'f':
		ps = 19
		p4vp = 5.2
		fs = float(name_splt[1][1:])
		pdp = float(name_splt[2])
		np_info = name_splt[3]
		thickness = name_splt[4]
		
	else:
		ps =  float(name_splt[1])
		p4vp = float(name_splt[2])
		fs = float(name_splt[3][1:])
		pdp = float(name_splt[4])
		np_info = name_splt[5]
		thickness = name_splt[6][:-2]

	im_size = image_size

	np_percent = float(np_info.split('v')[0])
	np_size = float(np_info.split('v')[1])

	# find dimensions and conversion factor for the image
	width, height = image_data.shape
	pxls_to_um = int(width/im_size)

	# Image: grayscale version of the original AFM image
	if save_image == True:
		plt.imshow(image_data, cmap='Greys')
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_original.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

	# find the sample periodicity and FWHM:
	period, fwhm = char_periodicity(image_data, width, height, ps, pxls_to_um) # period in inverse pixels
	period = float(width/period) # period in pixels
	
	period_nm = period*1000/(float(pxls_to_um))

	fwhm_inv_nm = fwhm/float(pxls_to_um)

	grain_size = 2*3.141592*0.93/fwhm_inv_nm

	binarized = binarize_data(image_data, period)

	binarized = tidy_bin(binarized, 3.14*(period/6)**2) # delete components smaller than a small dot

	binarized = 1-binarized

	block_ratio = sum(sum(binarized))/(width*height)

	# skeletonize the binary image:
	skel_light = sk_light(binarized)
	skel_dark = sk_dark(binarized)

	# sort the morphological elements into categories of "dots" and "lines":
	dots_light, lines_light, line_image_light, n_dots_light = dots_and_lines(skel_light, 1-binarized, period)
	dots_dark, lines_dark, line_image_dark, n_dots_dark = dots_and_lines(skel_dark, binarized, period)

	dot_pixels_light = sum(sum(dots_light))
	line_pixels_light = sum(sum(lines_light))

	dot_pixels_dark = sum(sum(dots_dark))
	line_pixels_dark = sum(sum(lines_dark))

	line_percent_light = line_pixels_light/(dot_pixels_light+line_pixels_light)
	line_percent_dark = line_pixels_dark/(dot_pixels_dark+line_pixels_dark)

	# if there's a decent area fraction of dot features, perform an extra step to avoid merging dots together
	if np.logical_and(line_percent_light<=line_percent_dark, line_percent_light < 0.95):
		binarized = separate_dots(1-binarized, skel_light, period, 1-block_ratio)
		binarized = 1-binarized # flip back after analysis

	if np.logical_and(line_percent_light>line_percent_dark, line_percent_dark < 0.95):
		binarized = separate_dots(binarized, skel_dark, period, block_ratio)

	binarized = tidy_bin(binarized, 3.14*(period*block_ratio/3)**2) # delete components smaller than a small dot

	# repeat periodicity calculation, in case nanoparticles or low contrast were skewing the results:

	period, fwhm = char_periodicity(binarized, width, height, ps, pxls_to_um) #period in inverse pixels

	period = float(width/period) # period in pixels

	# save a snapshot, 5 periods x 5 periods

	snapshot = image_data[int(width/2-5*period):int(width/2+5*period),int(height/2-5*period):int(height/2+5*period)]

	# Image: small snapshot of the morphology
	if save_image == True:

		plt.imshow(snapshot, cmap='Greys')
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_snapshot.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

	period_nm = period*1000/(float(pxls_to_um))

	fwhm_inv_nm = fwhm/float(pxls_to_um)

	block_ratio = sum(sum(binarized))/(width*height)

	# Image: binary image
	if save_image == True:

		plt.imshow(binarized, cmap='Greys')
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_binarized.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()	

	# skeletonize the binary image:
	skel_light = sk_light(binarized)
	skel_dark = sk_dark(binarized)

	# preliminary defect analysis, used for tidying up the skeleton: 
	junct_light, junct_dark, ends_light, ends_dark = connectivity_results(skel_light, skel_dark, width, height)

	skel_tidy_light = prune_skel(skel_light, junct_light, period/5)
	skel_tidy_dark = prune_skel(skel_dark, junct_dark, period/5)

	# sort the morphological elements into categories of "dots" and "lines":
	dots_light, lines_light, line_image_light, n_dots_light = dots_and_lines(skel_tidy_light, 1-binarized, period)
	dots_dark, lines_dark, line_image_dark, n_dots_dark = dots_and_lines(skel_tidy_dark, binarized, period)

	# Images: morphology maps of the light and dark domains
	if save_image == True:
		plt.imshow(lines_light*0.1 + dots_light*0.3, cmap='CMRmap_r', interpolation='nearest')
		plt.imshow(skel_tidy_light, cmap='Greys', alpha=0.2)
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_light_sorted.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

		plt.imshow(lines_dark*0.1 + dots_dark*0.3, cmap='CMRmap_r', interpolation='nearest')
		plt.imshow(skel_tidy_dark, cmap='Greys', alpha=0.2)
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_dark_sorted.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

	# calculate line and dot fractions for each phase
	dot_pixels_light = sum(sum(dots_light))
	line_pixels_light = sum(sum(lines_light))

	dot_pixels_dark = sum(sum(dots_dark))
	line_pixels_dark = sum(sum(lines_dark))

	line_percent_light = line_pixels_light/(dot_pixels_light+line_pixels_light)
	line_percent_dark = line_pixels_dark/(dot_pixels_dark+line_pixels_dark)

	# defect analysis: analyze the connectivity of the line elements
	junct_light, junct_dark, ends_light, ends_dark = connectivity_resultsII(line_image_light*skel_tidy_light, line_image_dark*skel_tidy_dark, width, height)

	# Images: defect maps of the PS and P4VP
	if save_image == True:

		# locations of defects as x and y coordinates:
		x_insct_light, y_insct_light = np.nonzero(junct_light.T)
		x_end_light, y_end_light = np.nonzero(ends_light.T)
		x_insct_dark, y_insct_dark = np.nonzero(junct_dark.T)
		x_end_dark, y_end_dark = np.nonzero(ends_dark.T)

		ms = period/4 #marker size scales with the period

		plt.imshow(binarized, cmap='Greys',interpolation='nearest', alpha=0.5)
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.scatter(x_end_light, y_end_light, marker='o', c='red', s=ms)
		plt.scatter(x_insct_light, y_insct_light, marker='s', c='black', s=ms)
		plt.axis('off')
		plt.savefig(results_folder+image_name+'_light_defects.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

		plt.imshow(binarized, cmap='Greys',interpolation='nearest', alpha=0.5)
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.scatter(x_end_dark, y_end_dark, marker='o', c='red', s=ms)
		plt.scatter(x_insct_dark, y_insct_dark, marker='s', c='black', s=ms)
		plt.axis('off')
		plt.savefig(results_folder+image_name+'_dark_defects.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

	# calculate defect densities
	insct_density_light = sum(sum(junct_light))/sum(sum(line_image_light))*(int(pxls_to_um))**2
	insct_density_dark = sum(sum(junct_dark))/sum(sum(line_image_dark))*(int(pxls_to_um))**2
	
	end_density_light = sum(sum(ends_light))/sum(sum(line_image_light))*(int(pxls_to_um))**2
	end_density_dark = sum(sum(ends_dark))/sum(sum(line_image_dark))*(int(pxls_to_um))**2

	# map out the pattern morphology, based on the labelled features (a score of 0-8, with 4 being all lines and no dots)
	morph_data = -1*dots_light/(1-block_ratio) + dots_dark/(block_ratio) 
	morph_map, new_bin = morphology_map(morph_data, binarized, width, height, period)
	mean_morphology = np.mean(morph_map)

	# Image: morphology map of the microdomains
	if save_image == True:	
		plt.imshow(morph_map, interpolation='nearest',cmap=twilight_trun)
		plt.clim(0, 7)
		plt.imshow(binarized, cmap=bin_cmap, alpha=0.9)
		plt.axis('off')
		cur_axes = plt.gca()
		cur_axes.axes.get_xaxis().set_visible(False)
		cur_axes.axes.get_yaxis().set_visible(False)
		plt.savefig(results_folder+image_name+'_morphology_map.png', dpi=500, bbox_inches='tight', pad_inches = 0)
		if show_image == True:
			plt.show()
		plt.clf()

	# finally, return all calculated parameters as a dictionary

	results = {'name': image_name, 'ps':ps, 'p4vp':p4vp, 'pdp':pdp, 'fs':fs, 'f_np':np_percent, 'np_size':np_size, 'thickness':thickness,
				'pxls_to_um':pxls_to_um,'img_size':im_size, 'periodicity':period_nm, 'grain_size':grain_size, 'mean_morph':mean_morphology,
				'percent_lines_dark':line_percent_dark,'percent_lines_light':line_percent_light,'block_ratio':block_ratio,
				'junct_density_dark':insct_density_dark,'end_density_dark':end_density_dark, 
				'junct_density_light':insct_density_light,'end_density_light':end_density_light}

	return results