# Spectral Neural Models 


## Overview

This repository contains notes and code relating to spectral- (i.e. frequency) domain models of macro-scale neural dynamics. 

The primary focus of this type of neurophysiological model is to reproduce two key features of the M/EEG power spectrum: 

i) Spectral peaks (alpha, theta, etc.); number, location, magnitude  

ii) Power law scaling exponents  


We are interested in three things:

a) *understanding* - derivation of, motivation for, and behaviour of various spectral-domain neural models. 

b) *simulating*  - using existing, modified, and novel models

c) *fitting*  to empirical M/EEG power spectra, and assessing alternative optimization techniques (`scipy.optimize`, `scikit-optimize`, `tensorflow`, `STAN`, etc.)


## Organization

`notes/`  - Technical descriptions and general reflections on models, data, and science
`code/`   - The beating heart. Core functions (mostly python code) for simulating power spectra and data fitting. 
`data/` - Empirical power spectrum recordings from various sources. 
`scratch/` - Miscellaneous and work-in-progress. Unapologetically messy. 

