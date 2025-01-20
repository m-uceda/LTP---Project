from exploratory_data_analysis import get_mapping, map_emojis, change_to_pandas, class_distribution, preprocess_es_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig
from datasets import Dataset, load_dataset, ClassLabel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple, List
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rcParams
from torch import nn
import seaborn as sns
import pandas as pd
import torch
import os

rcParams['font.family'] = 'Segoe UI Emoji'

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        all_labels = []
        for data in self.train_dataset:
            all_labels.append(data['label'])
        n_labels = len(set(all_labels))

        self.class_distribution = calculate_class_distribution(self.train_dataset, n_labels)
        total = sum(self.class_distribution)
        self.class_weights = [total / freq if freq > 0 else 0.0 for freq in self.class_distribution]

        # Normalize class weights
        max_weight = max(self.class_weights)
        self.class_weights = [weight / max_weight for weight in self.class_weights]
        self.class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure weights are on the correct device
        self.class_weights_tensor = self.class_weights_tensor.to(model.device)

        # Define loss function with class weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def load_train_and_evaluate(
        model_name: str, 
        dataset_name: str,
        num_classes: int,
        file_name: str,
        mapping_file: str,
        performance_message: str,
        trainer_type: str,
        train_subset_size: int = None,
        test_subset_size: int = None,
        spanish_data_prep: str = None
    ) -> None:
    """
    Loads a model and a dataset. Fine-tunes the model on the dataset and evaluates its performance.

    Args:
        model_name (str): Name of the model to be fine-tuned. 
        dataset_name (str): Name of the dataset.
        num_classes (int): Number of classes of the dataset.
        file_name (str): Name of the file in which the confusion matrix will be saved.
        mapping_file (str): Name of the file used for mapping the emojis.
        performance_message (str): Message that will precede the printed performance.
        trainer_type (str): Type of trainer to be used ('weighted' or 'standard').
        train_subset_size (int): Number of samples used for training.
        test_subset_size (int): Number of samples used for testing.
        spanish_data_prep (str): Type of preprocessing used for the Spanish dataset.
    """
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        num_labels=num_classes
    )

    # Split data in train, test
    train_dataset, test_dataset = load_and_split_dataset(dataset_name=dataset_name)
    if (not (train_subset_size == None)) and not (train_subset_size == None):
        subset_to_train = get_subset(train_dataset, size=train_subset_size)
        subset_to_test = get_subset(test_dataset, size=test_subset_size)
    else:
        subset_to_train = train_dataset
        subset_to_test = test_dataset

    if dataset_name == "guillermoruiz/MexEmojis":
        train_es_without_emojis, train_es_with_emojis = preprocess_es_data(subset_to_train)
        test_es_without_emojis, test_es_with_emojis = preprocess_es_data(subset_to_test)
    elif num_classes == 12:
        subset_to_train = reduce_num_classes(subset_to_train)
        subset_to_test = reduce_num_classes(subset_to_test)

    if spanish_data_prep == None:
        performance = train_and_get_performance(
            train_dataset=subset_to_train,
            test_dataset=subset_to_test,
            tokenizer=tokenizer, 
            num_classes=num_classes,
            model=model,
            file_name=file_name,
            mapping_file=mapping_file,
            trainer_type=trainer_type
        )
    elif spanish_data_prep == "not preprocessed":
        performance = train_and_get_performance(
            train_dataset=subset_to_train,
            test_dataset=subset_to_test,
            tokenizer=tokenizer, 
            num_classes=num_classes,
            model=model,
            file_name=file_name,
            mapping_file=mapping_file,
            trainer_type=trainer_type
        )
    elif spanish_data_prep == "with emojis":
        performance = train_and_get_performance(
            train_dataset=train_es_with_emojis,
            test_dataset=test_es_with_emojis,
            tokenizer=tokenizer, 
            num_classes=num_classes,
            model=model,
            file_name=file_name,
            mapping_file=mapping_file,
            trainer_type=trainer_type
        )
    elif spanish_data_prep == "without emojis":
        performance = train_and_get_performance(
            train_dataset=train_es_without_emojis,
            test_dataset=test_es_without_emojis,
            tokenizer=tokenizer, 
            num_classes=num_classes,
            model=model,
            file_name=file_name,
            mapping_file=mapping_file,
            trainer_type=trainer_type
        )

    with open("final_results.txt", 'a') as file:
        file.write(performance_message)
        file.write('\n')
        file.write(str(performance))
        file.write('\n\n')

def train_and_get_performance(
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer, 
        num_classes: int,
        model: AutoModelForSequenceClassification,
        file_name: str,
        mapping_file: str,
        trainer_type: str
    ):
    """
    Fine-tunes a model on a dataset and evaluates its performance.

    Args:
        train_dataset (Dataset): Dataset for the model to be trained with.
        test_dataset (Dataset): Dataset for the model to be tested on.
        tokenizer (AutoTokenizer):  Tokenizer to tokenize the data.
        num_classes (int): Number of classes of the dataset.
        model (AutoModelForSequenceClassification): Model to be fine-tuned.
        file_name (str): Name of the file in which the confusion matrix will be saved.
        mapping_file (str): Name of the file used for mapping the emojis.
        trainer_type (str): Type of trainer to be used ('weighted' or 'standard').
    
    Returns:
        Dictionary containing performance of the model on the test set.
    """
    # Tokenize data and extract validation set
    tokenized_train, tokenized_validate = tokenize(
        dataset=train_dataset,
        tokenizer=tokenizer,
        num_classes=num_classes,
        test_size=1/9)      

    # Fine tune model
    trainer = get_trainer(model, tokenizer, tokenized_train, tokenized_validate, trainer_type)
    trainer.train()

    folder_path = "confusion matrices (final results)"
    os.makedirs(folder_path, exist_ok=True)
    sav_path = os.path.join(folder_path, file_name)

    # Evaluate performance on test set
    performance = evaluate_performance(
        model=model, 
        tokenizer=tokenizer, 
        test_dataset=test_dataset,
        mapping_file=mapping_file,
        save_path=sav_path)

    return performance

def load_model_and_tokenizer(
        model_name: str, 
        num_labels: int
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model to load.
        num_labels (int): The number of labels (classes) of the dataset used.

    Returns:
        Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
            - tokenizer (AutoTokenizer): The tokenizer for the specified model.
            - model (AutoModelForSequenceClassification): The model to load.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=num_labels)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    return tokenizer, model

def load_and_split_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """
    Load a dataset and split it into training and testing datasets.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Tuple[Dataset, Dataset]:
            - train_dataset (Dataset): The training dataset.
            - test_dataset (Dataset): The testing dataset.
    """
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if dataset_name == "guillermoruiz/MexEmojis":
        train_dataset = train_dataset.rename_column("text", "sentence")
        test_dataset = test_dataset.rename_column("text", "sentence")

        es_mapping = "es_mapping.txt"
        emoji_mapping = get_mapping(es_mapping)

        valid_labels = [emoji_mapping[i][1] for i in range(len(emoji_mapping))]
        class_label = ClassLabel(names=valid_labels)

        train_dataset = train_dataset.cast_column("label", class_label)
        test_dataset = test_dataset.cast_column("label", class_label)

    return train_dataset, test_dataset

def tokenize(dataset: Dataset, tokenizer: AutoTokenizer, num_classes: int, test_size: int) -> Tuple[Dataset, Dataset]:
    """
    Tokenize a dataset and split into train and validation sets.

    Args:
        dataset (Dataset): The dataset to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing.
        num_classes (int): Number of classes of the used model.
        test_size (int): Size of the testing set.

    Returns:
        Tuple[Dataset, Dataset]:
            - train_dataset (Dataset): Tokenized training dataset.
            - validate_dataset (Dataset): Tokenized validation dataset.
    """
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    # Convert label column to ClassLabel
    if not isinstance(dataset.features["label"], ClassLabel):
        dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

    # Stratified split
    split = dataset.train_test_split(test_size=test_size, stratify_by_column='label')

    train_dataset = split["train"].map(preprocess_function, batched=True, batch_size=64)
    validate_dataset = split["test"].map(preprocess_function, batched=True, batch_size=64)

    return train_dataset, validate_dataset

def get_subset(dataset: Dataset, size: int) -> Dataset:
    """
    Gets a stratified subset of a dataset.

    Args:
        dataset (Dataset): Dataset from which a subset will be extracted.
        size (int): Size of the extracted subset.
    
    Returns:
        Subset of 'size' number of samples.
    """
    # Convert dataset to pandas DataFrame for stratification
    df = dataset.to_pandas()

    # Perform stratified sampling
    train_df, _ = train_test_split(
        df,
        train_size=size,
        stratify=df['label'], 
        random_state=42
    )

    return Dataset.from_pandas(train_df)

def get_trainer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenized_train: Dataset,
    tokenized_validate: Dataset,
    trainer_type: str
) -> Trainer:
    """
    Set up a Trainer for fine-tuning a model.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained model to fine-tune.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing.
        tokenized_train (Dataset): The tokenized training dataset.
        tokenized_validate (Dataset): The tokenized validation dataset.
        trainer_type (str): Type of trainer to be used ('weighted' or 'standard').

    Returns:
        Trainer: A Trainer object.
    """
    training_args = TrainingArguments(
        output_dir="./results",                 # Directory for model outputs
        eval_strategy="epoch",                  # Evaluate at the end of each epoch
        learning_rate=2e-5,                     # Learning rate
        per_device_train_batch_size=16,         # Batch size for training
        per_device_eval_batch_size=32,          # Batch size for evaluation
        num_train_epochs=2,                     # Number of training epochs
        weight_decay=0.01,                      # Weight decay for optimization
        save_strategy="epoch",                  # Save once per epoch
        save_total_limit=1                      # Keep only the latest checkpoint
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if trainer_type == "weighted":
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validate,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validate,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

    return trainer

def reduce_num_classes(dataset: Dataset) -> Dataset:
    """
    Filters the dataset to include only the top N classes.

    Args:
        dataset (Dataset): Dataset that needs its classes to be reduced.

    Returns:
        Filtered Dataset (with only 12 most abundant classes).
    """
    df = change_to_pandas(dataset)

    # Load emoji mapping
    en_mapping_path = "us_mapping.txt"
    emoji_mapping = get_mapping(en_mapping_path)

    # Filter dataset to retain only the top 12 classes
    filtered_df, top_classes = filter_top_classes(df, num_classes=12)
    print(f"Top classes: {top_classes}")

    # Visualize class distribution after filtering
    class_distribution(filtered_df, emoji_mapping, "English (Filtered)")

    # Convert filtered df back to Dataset
    filtered_df = filtered_df.reset_index(drop=True)
    return Dataset.from_pandas(filtered_df)

def filter_top_classes(df: pd.DataFrame, num_classes: int = 12) -> Tuple[pd.DataFrame, List[int]]:
    """
    Filters the dataset to include only the top num_classes.

    Args:
        df (pd.DataFrame): DataFrame with dataset.
        num_classes (int): Number of top classes to retain.

    Returns:
        Filtered DataFrame and list of top class labels.
    """
    # Get label distribution
    label_counts = df["label"].value_counts()
    # Identify top N classes
    top_classes = label_counts.index[:num_classes]
    # Filter dataset to keep only top N classes
    filtered_df = df[df["label"].isin(top_classes)].copy()
    # Map labels to new indices
    class_mapping = {label: idx for idx, label in enumerate(top_classes)}
    filtered_df["label"] = filtered_df["label"].map(class_mapping)

    return filtered_df, top_classes

def calculate_class_distribution(dataset: Dataset, num_classes: int, label_column: str = "label"):
    """
    Calculates the class distribution in a dataset.

    Args:
        dataset (Dataset): A HuggingFace Dataset object or any iterable with label data.
        label_column (str): The name of the column containing the labels.
        num_classes (int): The total number of classes.

    Returns:
        list: A list of length 'num_classes' where each element represents 
                the number of occurrences of the corresponding label.
    """
    # Extract the labels from the dataset
    labels = dataset[label_column]
    
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Create a list with counts for each class
    class_distribution = [label_counts.get(i, 0) for i in range(num_classes)]
    
    return class_distribution

def evaluate_performance(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        test_dataset: Dataset, 
        mapping_file: str, 
        save_path: str = "confusion_matrix.png"
    ) -> dict:
    """
    Evaluate the performance of a model on a test dataset using multiple metrics and plot a confusion matrix.

    Args:
        model (AutoModelForSequenceClassification): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing.
        test_dataset (Dataset): The test dataset.
        mapping_file (str): Name of the file used for mapping the emojis.
        save_path (str): Name of the file in which the confusion matrix will be saved.

    Returns:
        dict: A dictionary containing the computed metrics and their values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_labels = []
    all_predictions = []

    for data in test_dataset:
        input_string = data['sentence']
        label = data['label']

        inputs = tokenizer(input_string, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = torch.argmax(logits, dim=1).item()

        all_labels.append(label)
        all_predictions.append(predicted_label)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average="weighted"),
        "recall": recall_score(all_labels, all_predictions, average="weighted"),
        "f1": f1_score(all_labels, all_predictions, average="weighted")
    }

    df = change_to_pandas(test_dataset)
    emoji_mapping = get_mapping(mapping_file)
    class_count = df["label"].value_counts()
    em_labels = [map_emojis(emoji_mapping, label) for label in class_count.index]

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, em_labels, save_path=save_path)

    return metrics

def plot_confusion_matrix(
        all_labels: List, 
        all_predictions: List, 
        class_names: List, 
        save_path: str = None
    ) -> None:
    """
    Plot a confusion matrix for the model predictions.

    Args:
        all_labels (List): True labels of the test dataset.
        all_predictions (List): Predicted labels by the model.
        class_names (List): List of class names corresponding to the labels.
        save_path (str): Path to save the confusion matrix image.

    Returns:
        None
    """
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()