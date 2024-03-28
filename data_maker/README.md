# Data preparation

There are a couple of steps we take to prepare useful data.

|-Z_score.ipynb

This file contains code needed to z-score the voxel-wise responses within sessions and to average it across repeats of the same images. It also contains code needed to resize the COCO images using the crops used by the NSD experiment.

|-data_reorder.py

The output of Z_score.ipynb is not associated with a COCO id, and is very large as it contains the full brain volume. This file will extract out just the cortex data, and associate the COCO ids with the cortex data.

|-rank_by_HCP.ipynb

It turns out, the full cortical surface is still a lot of information. We obviously only want regions that respond consistently to visual stimuli. This file will take the noise ceiling information, take the Human Connectome Project
(HCP) region masks, and for each subject, rank how consistently each region responds to visual stimuli. We average the ranks across all eight subjects to get the top regions.


Run the files in the following order:
1. Z_score.ipynb. This will take in "raw" NSD data, and output z-scored 1D beta values, order of the neural data, and resized images
2. data_reorder.py. This will use the order and associate each beta value with an image name. It will also take the volume, and just preserve the cortical voxels.
3. rank_by_HCP.ipynb. This ranks the HCP regions by average noise ceiling's rank across eight subjects. Uses to reduce the information modeled. This is a simple filter. 


Note, that currently the code is written using hard coded paths, please modify them as needed. Or you can use the data that we have prepared for you, please download the data from the torrent provided on the main page.