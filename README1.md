To reproduce the results from this paper: Transfer learning for improved generalizability in causal
physics-informed neural networks for beam simulations

### Folder Structure

The data and code underlying the publication "Transfer learning for improved generalizability in causal physics-informed neural networks for beam simulations" are organized into seven main folders: Combine_plots, EB_main, Euler_transfer_Causal, Euler_transfer_PINN, Revision_Experiments, Timo, and TIMO_transfer. Below is a detailed description of each folder and its contents.

### Combine_plots

This folder contains code for PGNN (physics-guided neural networks) and gPINN (gradient-enhanced physics-informed neural networks). All implementations are done using Jupyter notebooks (.ipynb). To run the notebooks, simply execute the cells. The ".pth" files are the trained models.

### EB_main

This folder contains three subfolders of codes including comparison results of the main paper: Causal_PINN, SA_PINN, and PINN_EB. Each subfolder contains Jupyter notebooks (.ipynb). To run these, simply execute the cells. It also includes trained model files (.pkl, .pth), log files (.log) showing the results at every iteration, and (.sh) files to execute on the cluster (not needed).

### Euler_transfer_Causal

This folder contains four subfolders, each representing a different case for the Euler-Bernoulli model with Causal PINN transfer learning and noise case experiments. Each subfolder includes:

1. Jupyter notebooks (.ipynb) containing the code for experiments.
2. Trained model files (.pkl, .pth).
3. Log files (.log) showing the results at every iteration, and (.sh) files to execute on the cluster.

### Euler_transfer_PINN

This folder contains code for PINNs for solving the Euler-Bernoulli beam including noise experiments. It includes:
1. .ipynb files to reproduce figures.
2. A main.py file used on the Delft Blue cluster.

### Revision_Experiments

This folder contains five subfolders with comparison results and weights visualization implementations. All these subfolders contain Jupyter notebooks and .py Python files. To run .py files, execute "python3 file_name.py" on the command line.

### Timo

This folder contains three subfolders of codes including comparison results of the main paper: Causal_PINN, SA_PINN, and PINN for the Timoshenko beam model. Each subfolder contains Jupyter notebooks (.ipynb). To run these, simply execute the cells.

### TIMO_transfer

This folder contains test cases for space-time extension with transfer learning in Causal-PINN implementation. It also includes (.ipynb) notebooks; to run these notebooks, simply execute the cells.

### General Notes

- All codes, when run, can use the trained models (.pkl, .pth) to output the reported results.
- To retrain the networks, ensure you store old models elsewhere as the current code replaces the previously trained models and figures.
- For retraining, uncomment the training part containing the block "history" and comment out the load model line.
- The times-new-roman.ttf file is used for figure axis-label formatting in "Times New Roman" format.
