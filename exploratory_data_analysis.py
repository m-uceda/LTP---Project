from datasets import load_dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import re

# Define font for matplotlib to show the emojis.
rcParams['font.family'] = 'Segoe UI Emoji'


def load_data(file_path):
    """Loads a dataset from huggingface.com."""
    data = load_dataset(file_path)
    return data


def get_mapping(file_path):
    """Retrieves the mapping from the english data from the text file."""
    file = open(file_path, "r", encoding="utf-8").readlines()
    mapping = []
    for line in file:
        mapping.append(line.split())
    return np.array(mapping)


def map_emojis(mapping, label):
    """Using the mapping, returns an emoji corresponding to the integer label."""
    if isinstance(label, (int, np.integer)):
        return mapping[label][1]
    else:
        return label


def change_to_pandas(data):
    """Changes the train subset of a huggingface dataset doc to a pandas dataframe."""
    df = pd.DataFrame(data['train'])
    return df


def basic_info(df, language):
    """Prints the general info of a dataframe."""
    print(f"\n----- BASIC INFO {language.upper()} -----")
    print(df.info)
    print(df.columns)


def class_distribution(df, emoji_map, language):
    """Prints the class distribution of a dataframe as well as plots showing the instance counts by class."""
    print(f"\n----- CLASS DISTRIBUTION {language.upper()} -----")

    # Prints the percentages of the class distribution
    class_dist = df["label"].value_counts(normalize=True)
    for label, proportion in class_dist.items():
        print(f"{map_emojis(emoji_map, label)}: {proportion * 100:.2f}%")

    # Plots the instance count of each class
    class_count = df["label"].value_counts()
    em_labels = [map_emojis(emoji_map, label) for label in class_count.index]
    plt.barh(em_labels, class_count.values)
    plt.title(f'Class Distribution, {language}')
    plt.ylabel('Class')
    plt.xlabel('Instance Count')
    plt.show()


def character_count_by_class(df, emoji_map, language):
    """Prints information as well as box plots about the character count by class."""
    print(f"\n----- CHARACTER COUNT {language.upper()}-----")
    df['char_count'] = df['sentence'].apply(len)
    print(f"Range: {df['char_count'].min():.2f} to {df['char_count'].max():.2f}")
    print(
        f"Character Count mean = {df['char_count'].mean():.2f}, median = {df['char_count'].median()}, and standard deviation = {df['char_count'].std():.2f}")

    df_sorted = df.sort_values(by='label')
    groups = [group['char_count'].values for name, group in df_sorted.groupby('label')]
    sorted_labels = sorted(df['label'].unique())
    em_labels = [map_emojis(emoji_map, label) for label in sorted_labels]

    plt.boxplot(groups, labels=em_labels)
    plt.title(f'Character Count by Class, {language}')
    plt.xlabel('Class')
    plt.ylabel('Number of Characters')
    plt.show()


def word_count_by_class(df, emoji_map, language):
    """Prints information as well as box plots about the word count by class."""
    print(f"\n----- WORD COUNT {language.upper()}-----")
    df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
    print(f"Range: {df['word_count'].min():.2f} to {df['word_count'].max():.2f}")
    print(
        f"Word Count mean = {df['word_count'].mean():.2f}, median = {df['word_count'].median()}, and standard deviation = {df['word_count'].std():.2f}")

    df_sorted = df.sort_values(by='label')
    groups = [group['word_count'].values for name, group in df_sorted.groupby('label')]
    sorted_labels = sorted(df['label'].unique())
    em_labels = [map_emojis(emoji_map, label) for label in sorted_labels]

    plt.boxplot(groups, labels=em_labels)
    plt.title(f'Word Count by Class, {language}')
    plt.xlabel('Class')
    plt.ylabel('Number of Words')
    plt.show()


def plot_comparison(dataframes, titles, type):
    """Makes box plots to compare different dataframes in different categories."""
    plt.boxplot(dataframes, labels=titles)
    plt.title(f"{type} Count")
    plt.ylabel(f"Number of {type}s")
    plt.xlabel(f"Corpus")
    plt.show()


def individual_analysis(df, version, mapping):
    """Initiates all the functions which do analysis on one dataframe."""
    basic_info(df, version)
    class_distribution(df, mapping, version)
    character_count_by_class(df, mapping, version)
    word_count_by_class(df, mapping, version)


def exploratory_analysis(en_df, es_df_original, es_df_without_emojis, es_df_with_emojis):
    """Initiates the functions for the exploratory analysis."""
    # Get the US mapping
    en_mapping = "us_mapping.txt"
    emoji_mapping = get_mapping(en_mapping)

    # Individual Analysis
    individual_analysis(en_df, "English", emoji_mapping)
    individual_analysis(es_df_original, "Mexican Spanish Unprocessed", False)
    individual_analysis(es_df_without_emojis, "Mexican Spanish without Emojis", False)
    individual_analysis(es_df_with_emojis, "Mexican Spanish with Emojis", False)

    # Compare the different dataframes
    plot_comparison([en_df['word_count'], es_df_original['word_count']],
                       ["English", "Mexican Spanish Unprocessed"], "Word")
    plot_comparison([en_df['char_count'], es_df_original['char_count']],
                     ["English", "Mexican Spanish Unprocessed"], "Character")
    plot_comparison([en_df['word_count'], es_df_original['word_count'], es_df_without_emojis['word_count']],
                      ["English", "Mexican Unprocessed", "Mexican Spanish Processed"], "Word")
    plot_comparison([en_df['char_count'], es_df_original['char_count'], es_df_without_emojis['char_count']],
                      ["English", "Mexican Unprocessed", "Mexican Spanish Processed"], "Character")


def remove_emojis_regex(text):
    """Creates a regex to remove emojis and pictograms."""
    # Unicodes of various types of pictograms, etc.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"  
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF" 
        "\U00002702-\U000027B0" 
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_emojis(ds):
    """Removes all emojis out of a dataset"""
    ds["sentence"] = remove_emojis_regex(ds["sentence"])
    return ds


def remove_tags(ds):
    """Removes all tags from a dataset"""
    # Take away the location tag and take away placeholders for emoijs, URLs, etc.
    ds["sentence"] = re.sub("^.+ _GEO|_[a-zA-Z]{3}", "", ds["sentence"])

    # Make sure there are no double whitespaces or whitespaces in the beginning or end
    ds["sentence"] = re.sub(" {2,}", " ", ds["sentence"])
    ds["sentence"] = re.sub("^ | $", "", ds["sentence"])
    return ds


def preprocess_es_data(ds):
    """Creates the two additional versions of the Spanish dataset."""
    ds_without = ds
    # Remove emojis
    ds_without = ds_without.map(remove_emojis)
    # Remove placeholders
    ds = ds.map(remove_tags)
    ds_without = ds_without.map(remove_tags)
    return ds_without, ds


def main():
    """Main function"""
    # Loads Data
    en_data_path = "Karim-Gamal/SemEval-2018-Task-2-english-emojis"
    es_data_path = "guillermoruiz/MexEmojis"
    dataset_en = load_data(en_data_path)
    dataset_es = load_data(es_data_path)
    dataset_es = dataset_es.rename_column("text", "sentence")

    # Preprocessing Spanish
    dataset_es_without_emojis, dataset_es_with_emojis = preprocess_es_data(dataset_es)

    # Exploratory Analysis
    exploratory_analysis(change_to_pandas(dataset_en), change_to_pandas(dataset_es),
                         change_to_pandas(dataset_es_without_emojis), change_to_pandas(dataset_es_with_emojis))


if __name__ == '__main__':
    main()
