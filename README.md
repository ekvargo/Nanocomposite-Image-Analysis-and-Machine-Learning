# Nanocomposite-Image-Analysis-and-Machine-Learning

The code in this repository performs automated image analysis on micrographs of block copolymers or visually-similar structures (defined dark and light domains with a characteristic periodicity)

This code was developed to extract quantitative information from atomic force microscopy (AFM) images, as published [here](https://doi.org/10.1002/adma.202203168).

It was strongly inspired by the [work](https://doi.org/10.1371/journal.pone.0133088) of Jeffrey Murphy, Kenneth Harris, and Jillian Buriak. You can access their ImageJ plugin [here](https://github.com/MurphysLab/ADAblock).

## Files:
1. `Image_Analysis_Environment.yml`
   
   This file can be used to make a conda environment for the image analysis code. Instructions on how to do so are available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

2. `Run_Image_Analysis_Public_Version.py` 

    This script is used to interface with a folder of microscope images, run them one-by-one through the analysis code, and then export the results of the analysis to a .csv file. 
    The code includes the option to automatically split images into subimages, analyzed separately, to increase a dataset's size and quantify intra-image variations.
    
3.  `Image_Analysis_Public_Version.py`
    
    This first version of the image analysis code works on standard BCP images *without* featureless or disordered domains. It defines a `analyze_image` function which is fed into
    the `Run_Image_Analysis` code. The output of this function is a dictionary of measurements (periodicity, defect density, etc.)
    
4. `Image_Analysis_Ternary_Public_Version.py`

    This is a second, more complicated version of the image analysis code. It supports images *with* featureless or disordered domains. The image binarization fails on large areas 
    without features, introducing artifacts. Here, the image is instead "ternarized", so that featureless areas are separate from dark and light microdomains. 
    The code defines an alternative `analyze_image` function which is fed into the `Run_Image_Analysis` code. 
    The output of this function is a dictionary of measurements, including the featureless area fraction of the image.

If you have any questions or problems with this code, please submit an issue or email the author directly (evargo@berkeley.edu). Thank you for your interest in our work!
