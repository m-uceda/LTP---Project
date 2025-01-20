# Language Technology Practical Project Group 2

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Main Script](#main-script)
- [Code](#code)

## Overview
This repository hosts the code for the Language Technology Practical Project of Group 2. It includes code for data exploratory analysis, preprocessing, model fine-tuning, and model evaluation.

## Installation

1. **Open terminal**.

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```

3. **Install required packages**
    ```bash
    pip3 install -r requirements.txt
    ```

## Main

1. **Execute the main script**
    ```bash
    python3 main.py
    ```

2. **Evaluation metrics** are available in the final_results.txt file once the script executed. Additionally, several confusion matrices generated, and can be found in the folder under the name ```'confusion matrices'```. After running the code we organized the results obtained in the ```'baseline results'``` folder and the ```'final results'``` folder (where results for both the standard and the weighted models can be found).

## Code
Almost all code is implemeneted in the ```src``` directory. The directory contains a file named ```utils.py``` with all the necessary functions for the task:
- ```load_train_and_evaluate()```: Loads a model and a dataset. Fine-tunes the model on the dataset and evaluates its performance.
- ```train_and_get_performance()```: Fine-tunes a model on a dataset and evaluates its performance.
- ```load_model_and_tokenizer()```: Loads a pre-trained model and tokenizer.
- ```load_and_split_dataset()```: Loads a dataset and splits it into training and testing datasets.
- ```tokenize()```: Tokenizes a dataset and splits into train and validation sets.
- ```get_subset()```: Gets a stratified subset of a dataset.
- ```get_trainer()```: Sets up a Trainer for fine-tuning a model.
- ```reduce_num_classes()```: Filters the dataset to include only the top N classes.
- ```filter_top_classes()```: Filters the dataset to include only the top num_classes.
- ```calculate_class_distribution()```: Calculates the class distribution in a dataset.
- ```evaluate_performance()```: Evaluates the performance of a model on a test dataset using multiple metrics and plot a confusion matrix.
- ```plot_confusion_matrix()```: Plots a confusion matrix for the model predictions.

Apart from the ```main.py``` scrpit some other files can be found in this directory:
- The ```baseline.py``` file executes the baseline logistic regression system.
- The ```exploratory_data_analysis.py``` contains useful functions for analysing the data.
- ```es_mapping.txt``` and ```en_mapping.txt``` were used to get the appropriate number-emoji mapping.
- The ```script_job.sh``` script was used to run the code on Hábrók.
