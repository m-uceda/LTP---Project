from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from typing import Tuple, List
import matplotlib.pyplot as plt
from src.utils import plot_confusion_matrix, load_and_split_dataset, preprocess_es_data


def evaluate_performance(
        all_labels: List, 
        all_predictions: List, 
        save_path: str = "confusion_matrix.png"
    ) -> dict:
    """
    Evaluate the performance of a model on a test dataset using multiple metrics and plot a confusion matrix.

    Args:
        all_labels (List): List with all the true labels.
        all_predictions (List): List with all the predicted labels.
        save_path (str): Name of the file where the confusion matrix will be saved.

    Returns:
        dict: A dictionary containing the computed metrics and their values.
    """

    # Determine class names from the labels
    n_labels = len(set(all_labels))
    class_names = [str(i) for i in range(n_labels)]

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average="weighted"),
        "recall": recall_score(all_labels, all_predictions, average="weighted"),
        "f1": f1_score(all_labels, all_predictions, average="weighted")
    }

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path=save_path)

    return metrics

def run_baseline(
        dataset_name: str,
        file_name: str,
        performance_message: str,
        spanish_data_prep: str = None
) -> None:
    """
    Runs baseline linear Logistic Regression system, evaluates its performance and creates confusion matrix.
    
    Args:
        dataset_name (str): Name of the dataset to be used.
        file_name (str): Name of the file where the confusion matrix will be saved.
        performance_message (str): Message that will precede the printed performance.
        spanish_data_prep (str): Type of preprocessing for the Spanish dataset.
    """
    # Load dataset
    train_dataset, test_dataset = load_and_split_dataset(dataset_name)


    if dataset_name == "guillermoruiz/MexEmojis":
        train_es_without_emojis, train_es_with_emojis = preprocess_es_data(train_dataset)
        test_es_without_emojis, test_es_with_emojis = preprocess_es_data(test_dataset)

    if spanish_data_prep == "with emojis":
        train_dataset = train_es_with_emojis
        test_dataset = test_es_with_emojis
    elif spanish_data_prep == "without emojis":
        train_dataset = train_es_without_emojis
        test_dataset = test_es_without_emojis

    # Extract text and labels
    train_texts = [example["sentence"] for example in train_dataset]
    train_labels = [example["label"] for example in train_dataset]
    test_texts = [example["sentence"] for example in test_dataset]
    test_labels = [example["label"] for example in test_dataset]

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    # Make predictions
    predictions = clf.predict(X_test)

    # Evaluate the model
    metrics = evaluate_performance(test_labels, predictions, save_path=file_name)
    with open("baseline_results.txt", 'a') as file:
        file.write(performance_message)
        file.write('\n')
        file.write(str(metrics))
        file.write('\n\n')

def main():
    # English
    run_baseline(
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        file_name="CM_English_baseline.png",
        performance_message="Performance Baseline English"
    )

    # Spanish (no prep)
    run_baseline(
        dataset_name="guillermoruiz/MexEmojis",
        file_name="CM_Spanish_noprep_baseline.png",
        performance_message="Performance Baseline Spanish (no prep)"
    )

    # Spanish (with emojis)
    run_baseline(
        dataset_name="guillermoruiz/MexEmojis",
        file_name="CM_Spanish_withemojis_baseline.png",
        performance_message="Performance Baseline Spanish (with emojis)",
        spanish_data_prep="with emojis"
    )

    # Spanish (without emojis)
    run_baseline(
        dataset_name="guillermoruiz/MexEmojis",
        file_name="CM_Spanish_withoutemojis_baseline.png",
        performance_message="Performance Baseline Spanish (without emojis)",
        spanish_data_prep="without emojis"
    )

if __name__ == "__main__":
    main()