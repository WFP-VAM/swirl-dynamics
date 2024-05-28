# ML4AA code


* *demo.ipynb*

The `demo.ipynb` notebook corresponds to the training of the diffusion model and the observation of some results for the mnist dataset. It provides an introduction to the key functions we'll be using next. In our case, the conditional diffusion model is the one that fits our needs. 

* *data-preparation.ipynb*

The `data-preparation.ipynb` notebook is used to read training and test data. To do this, we define a bbox and use hip-analysis to read forecasts and chirps data. These are then aligned, standardized (or not, if you want to train the model without standardization) and separated into train/test. 

* *read_data.py*

The `read_data.py` file contains functions for reading training and test data. These are previously prepared in the `data-preparation.ipynb` notebook, but are read as tensorflow datasets in this file. 

* *run_diffusion_with_our_data.ipynb*

This notebook contains the skeleton of the workflow: from the configuration of the diffusion model parameters to the training of the model. 

* *inference.ipynb*

The `inference.ipynb` notebook lets you use the previously trained model to evaluate it on all test data. Also, at the end of the notebook we observe a time series corresponding to a given point. 

* *benchmarks.ipynb*

The `benchmarks.ipynb` notebook evaluates Quantile Mapping and Bilinear Interpolation methods on test data. This will enable us to compare the results of the diffusion model with these methods. 

* *evaluation.ipynb*

In the `evaluation.ipynb` notebook, we simply compare the results of the diffusion model with the results of the two benchmarks. Different metrics are then used, calculated first over time and then over space. 

* *data-exploration.ipynb*

The `data-exploration.ipynb` notebook has been created to quickly observe training data, for example the correlation between forecasts and chirps data. This notebook does not need to be run in order to train the diffusion model or evaluate the results.

* *compute-spi.ipynb*

In the `compute-spi.ipynb` notebook, we simply calculate the spi corresponding to the results of each method: diffusion, quantile mapping and bilinear interpolation. We also derive the spi from chirps. This will enable us to compare the spi (an anomaly) with the actual rainfall values. 