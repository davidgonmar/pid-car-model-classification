# Usage

First, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

To prepare the dataset, run the following command:

```bash
python -m scripts.prepare_dataset
```
This assumes you have Kaggle API credentials set up.

To check that the dataset has been prepared correctly, run the following command:

```bash
python -m scripts.check_dataset
```


To train the model, run the following command:

```bash
python -m scripts.train --experiment 1
```
