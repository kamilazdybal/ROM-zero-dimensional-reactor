# Reduced-order model for a zero-dimensional reactor

Code and materials that accompany the Chapter from my dissertation: *Reduced-order model for a zero-dimensional reactor*.

## Code

### Transport of PCA-derived manifold parameters: [`PC-transport.py`](code/PC-transport.py)

Running this script on your PC takes about 150 minutes. If you want much quicker results (of the order of 15 minutes), consider not running GPR (set `run_GPR = False`).

### Transport of regression-aware AE-derived manifold parameters: [`AE-transport.py`](code/AE-transport.py)

Running this script on your PC takes about 9 hours. If you want much quicker results (of the order of 15 minutes), consider not running GPR (set `run_GPR = False`).

#### Visualizing the results

This [Jupyter notebook](code/PC-transport-results.ipynb) can be used to upload and visualize the results of the [`PC-transport.py`](code/PC-transport.py) script.

This [Jupyter notebook](code/AE-transport-results.ipynb) can be used to upload and visualize the results of the [`AE-transport.py`](code/AE-transport.py) script.
