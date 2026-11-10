# Introduction to Neural Networks for 1D and 2D Regression

A practical introduction to training neural networks with PyTorch, developed by Adam Coxson for the MSc in Digital Chemistry at the University of Liverpool.

The workshop starts with fitting a one-dimensional function, extends the same ideas to a two-dimensional surface, and provides a starting script for a chemistry assignment using ZINDO energies and molecular fingerprints to predict TDDFT excitation energies.

## What you will learn

- Build feed-forward neural networks, from one hidden layer to a configurable multi-layer perceptron (MLP).
- Generate training data, normalise inputs and targets, and prepare PyTorch data loaders.
- Train with mean squared error loss and the AdamW optimiser.
- Evaluate predictions and investigate learning rate, batch size, network size, activation functions and training duration.
- Compare predictions with known functions using loss curves, heatmaps and surface plots.

Here, **1D** means one input feature, `x → y`, and **2D** means two input features, `(x, y) → z`. Both tasks predict a scalar output.

Some familiarity with Python and NumPy will help. The examples use the CPU; a GPU is not required. The regression data are generated in the code, so no external dataset is needed for the two initial notebooks.

## Repository contents

| Location | Contents |
| --- | --- |
| [workshop/](workshop/readme.md) | Notebooks, example scripts, network classes, plotting helpers and installation notes. |
| [1D notebook](workshop/MLP_for_1D_regression.ipynb) | Guided introduction, training-loop exercises and hyperparameter experiments. Start here. |
| [2D notebook](workshop/MLP_for_2D_regression.ipynb) | Regression with two input features, prediction heatmaps and 3D surface plots. |
| [docs/](docs/) | Workshop assignment brief, questions and solutions in Word format. |
| [Chemistry assignment starter](workshop/dml_submission.py) | Data-loading framework for the molecular prediction task. Requires separately supplied data and further implementation. |

## Installation
### 1. Get the files and open a terminal

Download and extract this repository. Alternatively, with Git installed:

```bash
git clone https://github.com/AdamCoxson/Intro-to-Neural-Nets-for-1D-and-2D-Regression.git
```

Install [Anaconda](https://www.anaconda.com/download) if you do not already have Conda.

- **Windows:** open **Anaconda Prompt** from the Start menu.
- **macOS:** open **Terminal** from Applications → Utilities, or a terminal in your editor, with Conda initialised.
- **Linux:** use a terminal with Conda initialised; the same environment setup below applies.

### 2. Create and activate the environment

```bash
conda create -n ml_workshop python=3.11
conda activate dmlworkshop
conda install -c conda-forge pip jupyter numpy matplotlib ipywidgets ipympl scipy scikit-learn ipykernel
```

### 3. Install PyTorch

For an installation suited to your operating system and hardware, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Select **Stable**, your operating system, **Pip** and **Python**; choose **CPU** on Windows/Linux for these exercises. On macOS, follow the Mac installation option. Run its command inside `ml_workshop`.

### 4. Additional packages

For the chemistry assignment, also install Pandas. RDKit is needed if you use the molecular descriptor helpers in `rdkit_functions.py`:

```bash
conda install -c conda-forge pandas rdkit
```

## Start the workshop

In the same terminal, navigate to the extracted or cloned repository, then:

```bash
cd Intro-to-Neural-Nets-for-1D-and-2D-Regression
cd workshop
jupyter notebook
```

If you are already inside the repository, start with `cd workshop`. If you downloaded a ZIP, its extracted folder may have `-main` appended to the name.

1. Open `MLP_for_1D_regression.ipynb` and select **Python (dmlworkshop)** as the kernel.
2. Before running it, remove or update the hard-coded `os.chdir("C:/Users/adamc/...")` line in its first code cell. If you launched Jupyter from `workshop/`, remove that line.
3. Run the cells in order, read the explanations and complete the exercises.
4. Continue with `MLP_for_2D_regression.ipynb`.

See the [workshop guide](workshop/readme.md) for the suggested learning sequence, standalone script commands, troubleshooting and chemistry assignment requirements.

## Chemistry assignment and supporting documents

After the regression exercises, read the [assignment brief](docs/ML-Workshop-dml-brief.docx) and use [`dml_submission.py`](workshop/dml_submission.py) as a starting point. This is an incomplete exercise framework: it loads molecular data and initialises a loss function, but you must implement the model, training and evaluation.

The required training and test CSV files are **not included** in this repository. Obtain them from the workshop organiser before starting this part; their expected filenames and columns are listed in the [workshop guide](workshop/readme.md#chemistry-assignment).

The [questions](docs/Training%20and%20testing%20a%20Neural%20Network%20Questions.docx) and [solutions](docs/Training%20and%20testing%20a%20Neural%20Network%20Questions%20-%20solutions.docx) are also available in `docs/`.
