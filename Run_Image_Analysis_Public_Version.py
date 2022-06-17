import numpy as np

import warnings
warnings.filterwarnings("ignore") # stop printing a precision warning

import math

import os
import csv
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pdb

params = {'legend.fontsize': 'x-large',
		 'axes.labelsize': 'x-large',
		 'axes.titlesize':'x-large',
		 'xtick.labelsize':'x-large',
		 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import scipy

import skimage
from skimage import color
import skimage.util
from skimage.transform import resize

from PIL import Image 
from Image_Analysis_Public_Version import analyze_image, crop_img


# Walk through the AFM image directory and find all images
jpgs = []

image_names = []
image_sizes = []

subimages = True # Perform the analysis on 4 subimages for every image, in addition to full-image analysis

data_file_location = "/Users/emmavargo/Desktop/test_set/test/"
results_file_location = "/Users/emmavargo/Desktop/results/"
data_csv_name = "image_analysis_output"


for root, dirs, files in os.walk(data_file_location):
	for file in files:
		if 'jpg' in file:
			print(file)
			im = Image.open(root + '/' + file)
			data = 1-color.rgb2gray(skimage.img_as_float(im))
			jpgs.append(data)
			image_names.append(file)
			file = file.replace('-','_') # use either a dash or an underscore to separate items
			name_split = file.split('_')
			

			if name_split[1][0] == 'f':
				image_sizes.append(float(name_split[5]))
				
			else:
				image_sizes.append(float(name_split[7])) # image size in microns

			im.close() #avoid a "too many files open" error

# Crop the scale bar, etc. out of every image

cropped_images = []

for img in jpgs:
	cropped_images.append(crop_img(img))

# Now have cropped_images and all_names, which match up

for i in range(len(cropped_images)):

	image_data = cropped_images[i]
	name = image_names[i]
	size = image_sizes[i]

	results_dict = analyze_image(image_data, name, size, True, False, results_file_location) # this is the analysis for the full image

	with open(r''+ data_csv_name +'.csv', 'a', newline='') as csvfile: 
	
		fieldnames = ['name','ps', 'p4vp','pdp','fs', 'f_np','np_size','thickness', 'pxls_to_um','img_size','periodicity',
						'grain_size', 'mean_morph','block_ratio',
						'percent_lines_dark','junct_density_dark','end_density_dark',
						'percent_lines_light', 'junct_density_light','end_density_light']
		
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow(results_dict)

	# use periodicity from the full image to crop into subimages

	if subimages == True:

		width, height = image_data.shape

		n_subimages = 2
		pxls_to_um = results_dict['pxls_to_um']

		if results_dict['periodicity'] != 'f':
			period = results_dict['periodicity']*pxls_to_um/1000
			periods_per_side = 20

			for n in range(n_subimages): # just stick with 4 corners for all images
				if n == 0:
					n_low = 0
					n_high = int(0 + period*periods_per_side)
				else:
					n_high = width
					n_low = int(width - period*periods_per_side)

				for m in range(n_subimages):
					if m == 0:
						m_low = 0
						m_high = int(0 + period*periods_per_side)
					else:
						m_high = width
						m_low = int(width - period*periods_per_side)

					subimage_data = image_data[n_low:n_high, m_low:m_high]

					sub_name = name+'_sub_'+str(n)+str(m)

					# find adjusted size using pxls_to_um
					sub_width, sub_height = subimage_data.shape

					# perform image analysis
					sub_results_dict = analyze_image(subimage_data, sub_name, sub_width/pxls_to_um, True, False, results_file_location) # this is the analysis for the sub image

					with open(r''+ data_csv_name +'.csv', 'a', newline='') as csvfile: 
						fieldnames = ['name','ps', 'p4vp','pdp','fs', 'f_np','np_size','thickness', 'pxls_to_um','img_size','periodicity',
						'grain_size', 'mean_morph','block_ratio',
						'percent_lines_dark','junct_density_dark','end_density_dark',
						'percent_lines_light', 'junct_density_light','end_density_light']
						
						writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
						writer.writerow(sub_results_dict)

# Note: change the fieldname values if using the ternary analysis
#		fieldnames = ['name','ps', 'p4vp','pdp','fs', 'f_np','np_size','thickness', 
# 		'pxls_to_um','img_size','periodicity', 'grain_size', 'block_ratio',
# 		'frac_featureless', 'frac_dark_domains', 'frac_light_domains', 
# 		'area_frac_featureless', 'area_frac_light_dots', 'area_frac_lines', 'area_frac_dark_dots',
# 		'percent_lines_dark','junct_density_dark','end_density_dark',
# 		'percent_lines_light', 'junct_density_light','end_density_light']


