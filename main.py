from src.utils import load_train_and_evaluate

def main():
    """The main method of this script."""
    trainer_type = 'standard'

    # English (entire dataset)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=20,
        file_name="CM_english.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English:",
        trainer_type=trainer_type
        )
    
    # English (only 12 classes and same amount of data as Spanish) 
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="Karim-Gamal/SemEval-2018-Task-2-english-emojis",
        num_classes=12,
        file_name="CM_english_subset_12classes.png",
        mapping_file="us_mapping.txt",
        performance_message="Performance English (subset 12 classes):",
        trainer_type=trainer_type,
        train_subset_size=90765,        
        test_subset_size=38914
        )

    # Spansih (not preprocessed)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_notprep.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish (not prep):",
        trainer_type=trainer_type,
        spanish_data_prep="not preprocessed"
        )

    # Spansih (preprocessed with emojis)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_emojis.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish (with emojis):",
        trainer_type=trainer_type,
        spanish_data_prep="with emojis"
        )

    # Spansih (preprocessed without emojis)
    load_train_and_evaluate(
        model_name="google-bert/bert-base-multilingual-cased", 
        dataset_name="guillermoruiz/MexEmojis",
        num_classes=12,
        file_name="CM_spanish_noemojis.png",
        mapping_file="es_mapping.txt",
        performance_message="Performance Spanish (no emojis):",
        trainer_type=trainer_type,
        spanish_data_prep="without emojis"
        )
    

if __name__ == "__main__":
    main()