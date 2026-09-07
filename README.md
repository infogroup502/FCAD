# Multi-Prototype-Guided Fuzzy Contrast Inference for Industrial Multisensor Signal Anomaly Detection

## Requirements
The recommended requirements for FCAD are specified as follows:
- arch==5.3
- hurst==0.0.5
- matplotlib==3.5
- numpy==1.23
- pandas==1.5
- scikit-learn==1.2
- scipy==1.9
- statsmodels==0.13
- torch==1.13
- tqdm==4.64
- tsfres==0.20

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches the local CUDA runtime. CPU execution is also supported.

## Data

The datasets can be obtained and put into the `dataset/` folder in the following way:
- FCAD supports anomaly detection for multivariate industrial and multisensor time series datasets.
- If you want to use your own dataset, please place your dataset files in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.
- CSV files with the same naming convention are also supported.

For example:
```text
dataset/Genesis/Genesis_train.npy
dataset/Genesis/Genesis_test.npy
dataset/Genesis/Genesis_test_label.npy
```

## Usage

1. Install Python 3.9 or newer and the required dependencies.
2. Download or prepare the datasets and place them under the `dataset/` folder.
3. To train and evaluate FCAD on the  dataset, run the following command:
```bash
python main.py
```
