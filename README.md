# Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models
### Andrew F. Luo, Margaret M. Henderson, Leila Wehbe*, Michael J. Tarr* (*Co-corresponding Authors)
#### NeurIPS 2023 (Oral)
This is the official code release for ***Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models*** (NeurIPS 2023 Oral).

[Paper link](https://arxiv.org/abs/2306.03089), [Project site](https://www.cs.cmu.edu/~afluo/BrainDiVE/)
![teaser](https://github.com/aluo-x/BrainDiVE/assets/15619682/05772692-e052-4c96-84d9-1ca704385e3c)

***TLDR:*** The purpose of our framework is to explore the cortical selectivity of different brain regions. Our framework allows you to synthesize naturalistic images predicted to activate parts of higher order visual regions. All you need is a differentiable brain encoder (image -> brain activity predictor). You can then visually examine these images, or run new fMRI experiments with them.

### Abstract
A long standing goal in neuroscience has been to elucidate the functional organization of the brain. Within higher visual cortex, functional accounts have remained relatively coarse, focusing on regions of interest (ROIs) and taking the form of selectivity for broad categories such as faces, places, bodies, food, or words. Because the identification of such ROIs has typically relied on manually assembled stimulus sets consisting of isolated objects in non-ecological contexts, exploring functional organization without robust a priori hypotheses has been challenging. To overcome these limitations, we introduce a data-driven approach in which we synthesize images predicted to activate a given brain region using paired natural images and fMRI recordings, bypassing the need for category-specific stimuli. Our approach -- Brain Diffusion for Visual Exploration ("BrainDiVE") -- builds on recent generative methods by combining large-scale diffusion models with brain-guided image synthesis. Validating our method, we demonstrate the ability to synthesize preferred images with appropriate semantic specificity for well-characterized category-selective ROIs. We then show that BrainDiVE can characterize differences between ROIs selective for the same high-level category. Finally we identify novel functional subdivisions within these ROIs, validated with behavioral data. These results advance our understanding of the fine-grained functional organization of human visual cortex, and provide well-specified constraints for further examination of cortical organization using hypothesis-driven methods.


### Codebase
* Requirements (in addition to usual python stack)
  * Numpy
  * h5py
  * nibabel (install from pip)
  * pycortex (install from pip) -- only for brain 3D visualization
  * PyTorch 2.0.1 or above
  * Hugginface Diffusers (and Transformers and accelerate)

Download required encoder weights here. Download other optional metadata here.


Notes:

1. We trained an encoder for both the early visual and higher visual, but in the paper only used the higher visual part.

2. Originally, our encoder + dataloader was for multiple subjects. There is some residual code for this. Please ignore it and use it in the single subject mode.

3. In fMRI terminology, encoder means a function that predicts brain activations from images (network takes in an image, and predicts the brain activation).

4. There is a small bug in the original NeurIPS files, which effectively reduced the guidance scale hyper-parameter by a small amount (10% to 20%) for different brain regions. This bug occured due to the different ROI definitions which resulted in some ROI voxels being assigned to early visual (which we further mask out, and thus do not affect the gradient).

5. This bug does not affect the results, but you should keep in mind to reduce the guidance scale when you use the minimal demo code (I recommend around 100 if you use the OpenCLIP ViT B variant, go around 150 if you use EVA02-CLIP).


Broadly the project consists of 5 major steps:
1. Pre-process the dataset (we use NSD), derive the cortical voxel and the good (low-noise) voxel subset. For us this mainly includes voxels that respond consistently to visual stimulus.
2. Train a voxel-wise encoder for each subject independently. We further evaluate the R^2 of the encoder on the test set (the ~1000 shared images across 8 subjects in NSD), but this is not necessary.
3. Use the encoder (energy function) gradients to drive the diffusion model.
4. Optionally re-rank using the encoder itself (DALL-E style), but using the encoder itself as the objective.
5. Evaluation of the generated images using CLIP 5-way classification and human studies done via prolific. 

The organization of the code is as such:
```
|- checkpoints (this is where we store the encoder checkpoints)
  |- S1 (subject 1 out of 8)
  |- S2 (subject 2 out of 8)
  *
  *
  *
|- data_maker (this is where we store the data pre-processing scripts for NSD)
  |- README.md (read it!)
  |- Z_score.ipynb (z-scores each voxel and packages it into a nice format)
  |- data_reorder.py (matches each image name with the corresponding brain activations/beta weights)
  |- rank_by_HCP.ipynb (we select regions of the brain that have low-noise for visual input)
|- encoder_dataloader.py (Pytorch dataloader for encoder training)
|- encoder_options.py (Options/file paths you can set for the encoder training)
|- train_encoder.py (Pretty standard Pytorch training loop)
|- eval_encoder.py (Runs the encoder on the held out test set of around 1000 images seen by all eight subjects)
|- brain_guide_pipeline.py (This is a diffusers pipeline which we test on *SD2.1 base*. )
|- run_brain_guidance_category.py (Runs the pipeline on category selective regions)
|- run_brain_guidance_ROI.py (Runs the pipeline on FFA/OFA)
|- run_brain_guidance_sub_ROI.py (Runs the pipeline on the two halves of OPA and Food region)
|- demo_brain_guidance_NeurIPS.ipynb (Use the pipeline as an all in one demo!)
|- demo_brain_guidance_minimal.ipynb (A cleaned up demo so it is clear what we are doing!)
```

If you look at the core minimal example. You need to set a function to the pipeline that takes as input an image and outputs a predicted brain activation scalar. See `brain_tweak` in the brain_guide_pipeline.
TODO: cleanup code

