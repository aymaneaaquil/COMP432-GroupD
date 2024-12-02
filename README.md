# Comp432-F24-GroupD

Comp 432 Project for Group D, Fall 2024

## Table of Contents

- [Team Members](#team-members)
- [Project Description](#project-description)
- [Dataset Links](#dataset-links)
- [Running the Code](#running-the-code)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Training and Validating the Model](#training-and-validating-the-model)
  - [Training the Model on the Colorectal Cancer Imagery Dataset](#training-the-model-on-the-colorectal-cancer-imagery-dataset)
  - [Validating the Model on the Colorectal Cancer Imagery Dataset](#validating-the-model-on-the-colorectal-cancer-imagery-dataset)
  - [Testing the Model on the Prostate Cancer Imagery Dataset and the Animal Faces Dataset](#testing-the-model-on-the-prostate-cancer-imagery-dataset-and-the-animal-faces-dataset)
  - [Classifying the Prostate Cancer Imagery Dataset and the Animal Faces Dataset](#classifying-the-prostate-cancer-imagery-dataset-and-the-animal-faces-dataset)
- [Testing the Model on the Sample Test Dataset](#testing-the-model-on-the-sample-test-dataset)

## Team Members

- Khashayar Azad - 40211574
- Yeprem Antranik - 40204291
- Aymane Aaquil - 40204788
- Pooya Abdolghader - 40002811
- Maegan Losloso - 40247291

## Project Description

In this project, the transferability of a Convolutional Neural Network (CNN) was investigated across three datasets:

1. Dataset containing Colorectal Cancer Imagery
2. Dataset containing Prostate Cancer Imagery
3. Dataset containing Animal Faces

A CNN was trained on the Colorectal Cancer Imagery dataset and its ability to extract features from the other datasets without retraining was evaluated. More specifically, the model was tested on the Prostate Cancer Imagery dataset and the Animal Faces dataset.

## Dataset Links

The datasets can be downloaded from the following links:

- [Colorectal Cancer Dataset](https://zenodo.org/records/1214456)
- [Prostate Cancer Dataset](https://zenodo.org/records/4789576)
- [Animal Faces Dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## Setting up the environment

### Prerequisites

- Python 3.8+ installed
- `pip` (Python package manager)
- Optional: `virtualenv` for isolated environments

### Setup Steps

1. **Create a Virtual Environment**

   ```bash
   python -m venv myenv
   ```

2. **Activate the Virtual Environment**

   - **Windows**:
     ```bash
     myenv\Scripts\activate
     ```
   - **Linux/Mac**:
     ```bash
     source myenv/bin/activate
     ```

3. **Install Jupyter Notebook**

   ```bash
   pip install jupyter
   ```

4. **Install Dependencies**  
   Using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Start Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

6. **Install Jupyter Kernel for Virtual Environment**

   ```bash
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
   ```

## Training and Validating the Model

The `main.ipynb` notebook contains the code for training, validating, and testing the model on the Colorectal Cancer Imagery dataset, as well as testing the model on the Prostate Cancer Imagery dataset and the Animal Faces dataset.

### Training the Model on the Colorectal Cancer Imagery Dataset

In order to train the model, the `TRAIN_MODEL` flag in the `main.ipynb` notebook should be set to `True`. This enables the model to be trained on the Colorectal Cancer Imagery dataset and saved to the `trained_models` folder under the name `ResNet-final.pth`.

- If the `LOAD_TRAINED_MODEL` flag is set to `True`, the model will be loaded from the `trained_models` instead.

### Validating the Model on the Colorectal Cancer Imagery Dataset

The remaining cells in the `main.ipynb` notebook are used to:

#### Validating the Model on the Colorectal Cancer Imagery Dataset

- validate the model on the Colorectal Cancer Imagery dataset.
- visualize the encoded features of the Colorectal Cancer Imagery dataset.

#### Testing the Model on the Prostate Cancer Imagery Dataset and the Animal Faces Dataset

- encode the feature of the Prostate Cancer Imagery dataset using the trained model as well as a pretrained model on ImageNet dataset.
- visualize the encoded features of the Prostate Cancer Imagery dataset.
- encode the feature of the Animal Faces dataset using the trained model as well as a pretrained model on ImageNet dataset.
- visualize the encoded features of the Animal Faces dataset.

#### Classifying the Prostate Cancer Imagery Dataset and the Animal Faces Dataset

- classify the Prostate Cancer Imagery dataset using SVM with the encoded features of the trained model and a pretrained model on ImageNet dataset.
- classify the Animal Faces dataset using SVM with the encoded features of the trained model and a pretrained model on ImageNet dataset.

## Testing the Model on the Sample Test Dataset

In order for ease of testing, the `test_model.ipynb` notebook contains the code for testing the trained model on the sample test data of 100 images from the Colorectal Cancer Imagery dataset.

The cells in the `test_model.ipynb` notebook:

- load the trained model from the `trained_models` folder.
- encode the features of the sample test dataset and perform classification on the encoded features.
- assess the performance of the model using a confusiong matrix and a classification report.
