# Imaging settings

## Objective:
The Olympus UPlanSApo 60x oil objective provides high resolution with a working distance of 0.15 M.M. and a 1.35 N.A.
## Oil immersion: 1.518

# Channel information
| Channel Name | Fluorophore | Excitation (nm) | Emission (nm) | Dichroic (nm) | Organelle |
|--------------|-------------|-----------------|---------------|----------------|------------|
| Hoechst     | Hoechst 33342 | 361             | 486           | 405            | Nucleus    |
| Concanavalin A | Concanavalin A Alexa Fluor 488 | 495             | 519           | 488            | Endoplasmic Reticulum |
| WGA        | WGA Alexa Fluor 555 | 555             | 580           | 555            | Golgi Apparatus, Plasma Membrane |
| Phalloidin | Phalloidin Alexa Fluor 568 | 578             | 600           | 555            | F-actin    |
| MitoTracker | MitoTracker Deep Red  | 644             | 665           | 640            | Mitochondria |

## Deconvolution settings
The deconvolution files used can be found in the `./1.huygens_workflow_files` folder.

The settings in the Huygens software were as follows:
| Parameter | Value |
|-----------|-------|
| Algorithm | Classic Maximum Likelihood Estimation (CMLE) |
| PSF mode | Theoretical |
| Max. iterations | 30 |
| iteration mode | Optimized |
| Quality change threshold | 0.01 |
| Signal to noise ratio | 26 |
| Anisotropy mode | Off |
| Acuity mode | On |
| Background mode | Lowest value |
| Background estimation radius | 0.7 |
| Relative background | 0.0 |
| Bleaching correction | Off |
| Brick mode | Auto |
| PSFs per brick mode | Off |
| PSFs per brick | 1 |
| Array detector reconstruction mode | Auto |
