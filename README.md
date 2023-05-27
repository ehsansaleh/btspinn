# Learning PINNs with Integral Losses

This repository contains the code for the *"Learning from Integral Losses in Physics Informed Neural Networks"* paper.

In short, this paper pinpoints the biased nature of the MSE loss when training PINNs under integro-differntial equations, proposes multiple solution, and extensively benchmarks them on Poisson, Maxwell, and Smoluchowski PDE systems.

# Interactive Data Visualization Dashboards

You can check the following interactive dashboards for the ablation studies in the paper. 

* [The 2-Dimensional Poisson Problem Ablations Dashboard](https://ehsansaleh.web.illinois.edu/data/btspinn/01_poisson.html)
* [The Smoluchowski Problem Ablations Dashboard](https://ehsansaleh.web.illinois.edu/data/btspinn/02_smoluchowski.html)
* [The High-Dimensional Poisson Solutions Visualization Dashboard](https://ehsansaleh.web.illinois.edu/data/btspinn/06_poisshidim.html)
* [The High-Dimensional Poisson Training Curves Dashboard](https://ehsansaleh.web.illinois.edu/data/btspinn/07_hdpviz.html)

*Notes*:
  * These dashboards are standalone files with the data embedded in them, so it may take a few moments for them to load. 
  * If the dashboard layout was not properly organized, please zoom in/out the web page until a 

# Technical Information

1. **Python Environment**: Run `make venv` in a terminal to setup a virtual environment.

2. **Hyper-parameters**: The training configuration files, in JSON format, can be found at the `configs` directory.

3. **Running**: See `./main.sh` for an example bash script running the trainings.

4. **Training Code**: Check the [`bspinn/poisson.py`](./bspinn/poisson.py), [`bspinn/smoluchowski.py`](./bspinn/smoluchowski.py), and [`bspinn/maxwell.py`](./bspinn/maxwell.py) files.
