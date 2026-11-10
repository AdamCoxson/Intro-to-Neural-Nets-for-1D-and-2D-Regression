# Workshop guide

Work through the 1D notebook first, then extend the same training workflow to two input features in the 2D notebook. The Python scripts provide reference implementations and a starting point for further experiments.

## Before you start

Follow the [installation instructions in the main README](../README.md#installation). The original platform-specific notes are [`install.txt`](install.txt) for Windows and [`install_mac.txt`](install_mac.txt) for macOS.

Activate the environment, navigate to this `workshop/` directory and launch Jupyter:

```bash
conda activate dmlworkshop
jupyter notebook
```

Select **Python (dmlworkshop)** as the notebook kernel.

Keep the notebooks and helper `.py` files together in this directory. They import local modules directly, so the working directory matters. In the **1D notebook's first code cell**, remove the author-specific `os.chdir("C:/Users/adamc/...")` line if you launched Jupyter here, or replace it with the absolute path to your own `workshop/` directory.

To check the notebook's working directory:

```python
from pathlib import Path
print(Path.cwd())
print(Path("stats_and_plot_functions.py").exists())
```

The second line should print `True`.

## 1. Fit a 1D function

Open [`MLP_for_1D_regression.ipynb`](MLP_for_1D_regression.ipynb).

The notebook generates data from a known function, fits a neural network to randomly sampled training points and compares its predictions with regularly sampled validation points. It introduces:

- Networks with one and two hidden layers.
- Input and target normalisation using training-data statistics.
- Mini-batch training with `TensorDataset`, `DataLoader`, MSE loss and AdamW.
- Prediction plots and the difference between training, validation and testing.

Read and run the cells from top to bottom with **Shift+Enter**. The first demonstration uses 25 epochs. Later cells provide a framework for you to modify; some requested functionality is deliberately left for you to implement.

For the exercises, compare network architectures, add validation inside the training loop, plot both losses and experiment with hyperparameters. The notebook suggests spending up to 40 minutes on these tasks in class, prioritising tasks 2–5.

For best-model tracking, preserve an independent copy of the weights:

```python
from copy import deepcopy

# Inside the condition that checks for a new lowest validation loss:
best_model_state = deepcopy(model.state_dict())

# After training, before computing final predictions:
model.load_state_dict(best_model_state)
model.eval()
```

The existing examples use `model.state_dict().copy()`, which only copies the dictionary and does not preserve independent tensor values. Use `deepcopy` when completing this exercise.

## 2. Fit a 2D surface

Open [`MLP_for_2D_regression.ipynb`](MLP_for_2D_regression.ipynb).

Here the model takes two input features, `x` and `y`, and predicts one output, `z`. The code samples training points from a known surface and evaluates on a 100 × 100 grid.

Run the cells in order and examine:

- How the two normalised input columns are joined into a single input tensor.
- How changing network size, activation or training settings affects the fit.
- The true and predicted heatmaps, including their colourbar ranges.
- The prediction-minus-truth deviation map and the 3D surface plots.

The notebook suggests 15–30 minutes for this section during the workshop. Keep `dropout=0.0` initially, as the teaching code flags nonzero dropout for further checking.

Losses used during training are calculated on normalised targets. The 2D notebook also calculates a final MSE on the original target scale; these numbers are not directly comparable. If you change the best-model tracking, restore the weights **before** computing the final predictions and plots.

For a fair comparison between runs, change one setting at a time and record the settings, validation loss and observations. Reinitialise the model and optimiser for each experiment, and rerun dependent preprocessing cells when changing the data. Restart the kernel and run from the top if the notebook state becomes unclear.

The examples sample validation points from the same known function and domain as training. For a real dataset, retain a separate test set and fit preprocessing only on training data. NumPy sampling also varies between runs unless you set a NumPy seed; `torch.manual_seed` alone does not seed NumPy.

## Reference scripts and helper modules

| File | Purpose |
| --- | --- |
| [`network_models.py`](network_models.py) | Standalone 1D training example with validation tracking and prediction plots. |
| [`network_models_2D.py`](network_models_2D.py) | Standalone 2D training example with heatmaps and surface plots. |
| [`network_classes.py`](network_classes.py) | `SingleLayerPerceptron`, `MLP_2layer` and configurable `MLP` classes. |
| [`stats_and_plot_functions.py`](stats_and_plot_functions.py) | Sampling, known functions, scaling, metrics and plotting utilities. Imports scikit-learn as well as NumPy, SciPy and Matplotlib. |
| [`rdkit_functions.py`](rdkit_functions.py) | Optional molecular descriptor and column-category helpers for chemistry work. |
| [`dml_submission.py`](dml_submission.py) | Incomplete starter for the chemistry assignment, requiring separate CSV data. |
| [`loss_plot.png`](loss_plot.png) | Illustrative loss plot used in the 1D notebook. |

To run the reference scripts instead of using Jupyter, execute either command from this directory:

```bash
python network_models.py
python network_models_2D.py
```

Run them individually. They train immediately with settings defined in the source and display Matplotlib plots; they do not take command-line options. Close plot windows to let execution continue where needed. The default scripts use 200 epochs for 1D and 100 for 2D, so reduce `num_epochs` for an initial check. Edit `learning_rate`, `batch_size`, `neurons`, `activation` and `training_samples` to experiment.

The scripts also use the shallow best-state copy described above. Treat them as teaching references when implementing reliable checkpoint selection.

## Chemistry assignment

Read [`ML-Workshop-dml-brief.docx`](../docs/ML-Workshop-dml-brief.docx), then adapt [`dml_submission.py`](dml_submission.py) using what you learned from the notebooks. It prepares ZINDO energies and molecular fingerprints for a task involving TDDFT S1 excitation energies.

**This part needs data supplied separately by the workshop organiser.** The repository does not contain the CSV files or a completed training pipeline.

Create a `data/` folder inside `workshop/` and place the supplied files there:

| Expected path, relative to `workshop/` | Columns read by the starter |
| --- | --- |
| `data/train_data_10k_conjugated_cores.csv` | `S1_TDDFT`, `S1_ZINDO`, `fingerprint` |
| `data/test_data_500_analogues.csv` | `E(S1_TDDFT)_analogue`, `E(S1_ZINDO)_analogue`, `fingerprint` |

The `fingerprint` values must be string representations of Python lists that can be parsed by `ast.literal_eval`, with a consistent length.

Install Pandas for the starter and RDKit if using the descriptor helper:

```bash
conda install -c conda-forge pandas rdkit
```

Run your completed script from `workshop/` so its `os.getcwd()`-based data paths resolve correctly. The starter initially loads only 100 training rows with `num=100` for debugging; set `num=None` to load the full training file. Build a training/validation split from the training data and reserve the analogue test set for final evaluation.

The [questions](../docs/Training%20and%20testing%20a%20Neural%20Network%20Questions.docx) and [solutions](../docs/Training%20and%20testing%20a%20Neural%20Network%20Questions%20-%20solutions.docx) provide additional supporting material.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `conda` is not recognised | On Windows, use Anaconda Prompt. On macOS/Linux, check that Conda is installed and initialised for your shell, then reopen the terminal. |
| `ModuleNotFoundError` for `torch`, `sklearn` or another package | Activate `dmlworkshop`, install the missing package there and select its kernel in Jupyter. The package for `sklearn` is named `scikit-learn`. |
| Local modules cannot be imported | Check the working directory and keep all workshop helper files together. Remove or update the 1D notebook's hard-coded `os.chdir` path. |
| Interactive plots fail or remain blank | Check that `ipympl` and `ipywidgets` are installed in the selected environment. Restart the kernel; use `%matplotlib inline` in place of `%matplotlib ipympl` for static plots. Notebook magic commands belong in notebook cells, not ordinary Python scripts. |
| Conda stalls while solving | The original guides suggest trying an unqualified package name or the `conda-forge` channel. The main README provides a combined installation using `conda-forge`; check the actual solver error before changing packages. |
| Training takes too long | Lower `num_epochs`, reduce the number of training samples or use a smaller network for initial experiments. |
| Sampling raises a “larger sample than population” error | Sampling is without replacement. Reduce `training_samples` or decrease the grid spacing (`dx` or `dxdy`) to provide more candidate values. In 2D, the helper samples each coordinate without replacement independently. |
| Chemistry CSV files cannot be found | Obtain the separate datasets and put them in `workshop/data/`; launch the script with `workshop/` as the working directory. |
| Appendix images do not display | The 1D appendix refers to `SLP_run.png` and `2layer_run.png`, which are not included. This does not prevent the exercises from running. |
