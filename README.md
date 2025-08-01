﻿# Car Model Classification

This code is part of a project developed for the course "Procesamiento de Imágenes Digitales" (Digital Image Processing) at the University of Seville. We trained a ResNet model on an augmented version of the Stanford Cars dataset with Spanish car models.

## Usage

### Training

First, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

To prepare the dataset, run the following command:

```bash
python scripts.prepare_dataset
```

This assumes you have Kaggle API credentials set up. In order to set them up,

1. Create an account on Kaggle if you don't have one (<https://www.kaggle.com>)
2. Go to your profile → Account → Create New API Token
3. You will download a `kaggle.json` file
4. Create a directory `.kaggle` in the project root:
```bash
mkdir .kaggle
```
5. Move the `kaggle.json` file to the `.kaggle` directory:
```bash
mv /path/to/kaggle.json .kaggle/
```


To check that the dataset has been prepared correctly, run the following command:

```bash
python scripts.check_dataset
```

You can launch a training session with the following command:

```bash
python -m scripts.train --experiment <exp_name>
```

The details about experiments can be found in `lib/experiment.py`.

## Inference with Streamlit
You must have a trained checkpoint to run inference. You can run the following command to start a Streamlit app for inference:

```bash
streamlit run app.py -- <resnet_number> <trained_path>
```
For instance, you can run the following command to start a Streamlit app for inference with ResNet-18:

```bash
streamlit run app.py -- 18 ./resnet18.pth
```


where `<resnet_number>` is the number of the ResNet model you want to use (currently, only 18 and 50 are supported) and `<trained_path>` is the path to the trained checkpoint file.
