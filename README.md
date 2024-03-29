# biobank-scale-methods-paper-2023
Public repository accompanying the publication "Biobank-scale methods and projections for sparse polygenic prediction from machine learning", containing code and predictors.

Predictor files themselves can be found on [DropBox](https://www.dropbox.com/sh/i4z56zucoqbwsdb/AACyyOyPraTgx7K_lq47A-mla?dl=0'.)

Current code samples:

(1) fitting_share.ipynb : this is a jupyter/ipython notebook file that gives examples of how to run the fitting routine and cholesky band generation used in the analysis. Training files need to be generated separately and paths need to be set to run this code, but the commenting should allow other users to adapt the methods to their needs.

(2) ml-single.py : this is a python script file to do the lasso, enet, and l1-logistic machine learning done for the manuscript. It requires phenotype files, sets, and gwas to already be performed. This file requires two auxiliary files -- 'names_in-pipeline.py' and 'pipeline_utilities.py' -- that have non-public directry and data sources in them. Because of this ml-single.py by itself isn't able to be run, but the code gives enough detail that it should be reproducible with minor changes. These changes mostly have to do with how genotype files are loaded into memory.

More code samples coming soon
