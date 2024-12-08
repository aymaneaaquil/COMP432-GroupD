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
- [Testing the Model on the Sample Test Dataset](#testing-the-model-on-the-sample-test-dataset)

## Team Members

- Khashayar Azad - 40211574
- Yeprem Antranik - 40204291
- Aymane Aaquil - 40204788
- Pooya Abdolghader - 40002811
- Maegan Losloso - 40247291

## Project Description

In this project, the transferability of a Convolutional Neural Network (CNN) was investigated across three datasets:

1. Colorectal Cancer Imagery
2. Prostate Cancer Imagery
3. Animal Faces

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

- Validate the model on the Colorectal Cancer Imagery dataset
- Visualize the encoded features of the Colorectal Cancer Imagery dataset
- Compare the performance of the trained encoder and the pretrained ImageNet encoder on the Prostate Cancer and Animal Faces datasets
- Classify the Prostate Cancer and Animal Faces datasets using SVM with features from the pretrained encoder.

### Testing the Model on the Sample Test Dataset

The `test_model.ipynb` notebook contains the code for testing the trained model on the sample test dataset which contains 100 images from each class of the Colorectal Cancer, Prostate Cancer, and Animal Faces datasets. The code performs the following steps:

- Loading the trained model
- Encoding the features of the sample test dataset and performing classification on the encoded features
- Assessing the performance of the model using a confusion matrix and a classification report

Additionaly, if it is required to run the `main.ipynb` notebook on the sample test dataset, the following few lines need to be uncommented:

```python
# uncomment the following lines to use the sample test dataset
# dataset_colorectal_cancer_path = "./datasets/sample_test_dataset/Colorectal Cancer"
# dataset_prostate_path = "./datasets/sample_test_dataset/Prostate Cancer"
# dataset_animal_path = "./datasets/sample_test_dataset/Animal Faces"
```
