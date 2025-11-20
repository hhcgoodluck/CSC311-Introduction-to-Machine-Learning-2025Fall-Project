import os
import json
import re
import numpy as np
import pandas as pd

# Base directory = folder where this file lives
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the saved NB model, vocab and config
_PARAMS_PATH = os.path.join(_BASE_DIR, "nb_final_2_params.npz")
_VOCAB_PATH  = os.path.join(_BASE_DIR, "nb_final_2_vocab.json")
_CONFIG_PATH = os.path.join(_BASE_DIR, "nb_final_2_config.json")


def _load_model_and_config():
    """
    Load Multinomial Naive Bayes parameters, vocabulary, and configuration.

    From nb_final_1_params.npz we load:
        class_log_prior : np.ndarray of shape (n_classes,)
        feature_log_prob: np.ndarray of shape (n_classes, n_features)
        labels          : np.ndarray of shape (n_classes,)
        alpha           : np.ndarray of shape (1,)  (optional, only for reference)

    From nb_final_1_vocab.json we load:
        vocab : dict[str, int]. Mapping from token -> column index in the
                text feature block.

    From nb_final_1_config.json we load:
        config : dict
            - rating_cols
            - neutral_rating_value
            - best_tasks_col
            - subopt_tasks_col
            - target_tasks
            - text_cols
            - n_structured_features
            - max_text_features
    """
    params = np.load(_PARAMS_PATH, allow_pickle=True)

    class_log_prior = params["class_log_prior"]
    feature_log_prob = params["feature_log_prob"]
    labels = params["labels"]

    # alpha is only for reference, prediction doesn't need it
    alpha = params["alpha"][0] if "alpha" in params.files else None

    with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return class_log_prior, feature_log_prob, labels, alpha, vocab, config


# Helper regex patterns
_RATING_REGEX = re.compile(r"^(\d+)")          # capture leading integer
_TOKEN_PATTERN = re.compile(r"\b\w\w+\b")      # mimic CountVectorizer's token_pattern


def _extract_rating(response):
    """Extract leading integer from a Likert-style response."""
    match = _RATING_REGEX.match(str(response))
    return int(match.group(1)) if match else None


def _build_rating_matrix(df, rating_cols, neutral_value):
    """
    Build numeric rating matrix from multiple columns.

    Returns:
        X_ratings : (n_samples, len(rating_cols))
    """
    cols = []
    for col in rating_cols:
        raw = df[col].apply(_extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        cols.append(filled.to_numpy().reshape(-1, 1))
    return np.hstack(cols)


def _process_multiselect(series, target_tasks):
    """
    Turn multi-select free-text responses into list-of-tasks per row.
    """
    processed = []
    for response in series:
        if pd.isna(response) or response == "":
            processed.append([])
        else:
            text = str(response)
            present = [task for task in target_tasks if task in text]
            processed.append(present)
    return processed


def _multiselect_to_binary(processed_lists, target_tasks_sorted):
    """
    Convert lists of tasks into a binary matrix using a FIXED, SORTED task order.

    This mimics sklearn's MultiLabelBinarizer default behavior where columns
    are in sorted label order.
    """
    n_samples = len(processed_lists)
    n_tasks = len(target_tasks_sorted)
    X_multi = np.zeros((n_samples, n_tasks), dtype=np.int64)

    task_to_idx = {task: i for i, task in enumerate(target_tasks_sorted)}

    for i, task_list in enumerate(processed_lists):
        for task in task_list:
            j = task_to_idx.get(task)
            if j is not None:
                X_multi[i, j] = 1
    return X_multi


def _build_text_series(df, text_cols):
    """
    Concatenate multiple free-text columns into a single string per row.
    """
    combined = df[text_cols[0]].fillna("")
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna("")
    return combined


def _build_bow_matrix(text_series, vocab, max_text_features):
    """
    Build bag-of-words count matrix using the saved vocabulary.

    Parameters:
        text_series : pd.Series of str
        vocab : dict[str, int] mapping token -> index
        max_text_features : int

    Returns:
        X_text : np.ndarray of shape (n_samples, max_text_features)
    """
    n_samples = text_series.shape[0]
    X_text = np.zeros((n_samples, max_text_features), dtype=np.float32)

    for i, text in enumerate(text_series):
        if pd.isna(text):
            text = ""
        text = str(text).lower()

        # Tokenization similar to CountVectorizer default
        for match in _TOKEN_PATTERN.finditer(text):
            token = match.group(0)
            idx = vocab.get(token)
            if idx is not None and 0 <= idx < max_text_features:
                X_text[i, idx] += 1.0  # count occurrences

    return X_text


def predict_all(csv_path):
    """
    Run prediction for all rows in a given CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV with the same columns as training_data_clean.csv.

    Returns
    -------
    list[str]
        Predicted model label for each row (e.g., "ChatGPT", "Claude", ...).
    """
    (class_log_prior,
     feature_log_prob,
     labels,
     alpha,
     vocab,
     config) = _load_model_and_config()

    rating_cols   = config["rating_cols"]
    neutral_value = config["neutral_rating_value"]
    best_col      = config["best_tasks_col"]
    subopt_col    = config["subopt_tasks_col"]
    target_tasks  = config["target_tasks"]
    text_cols     = config["text_cols"]
    n_structured  = int(config["n_structured_features"])
    max_text_features = int(config["max_text_features"])

    # MultiLabelBinarizer uses sorted label order by default, so we do the same.
    target_tasks_sorted = sorted(target_tasks)

    df = pd.read_csv(csv_path)

    # Structured features
    X_ratings = _build_rating_matrix(df, rating_cols, neutral_value)

    best_lists = _process_multiselect(df[best_col], target_tasks)
    subopt_lists = _process_multiselect(df[subopt_col], target_tasks)

    X_best = _multiselect_to_binary(best_lists, target_tasks_sorted)
    X_subopt = _multiselect_to_binary(subopt_lists, target_tasks_sorted)

    X_structured = np.hstack([X_ratings, X_best, X_subopt])
    assert X_structured.shape[1] == n_structured, \
        f"Expected {n_structured} structured features, got {X_structured.shape[1]}"

    # Text features
    text_series = _build_text_series(df, text_cols)
    X_text = _build_bow_matrix(text_series, vocab, max_text_features)

    # Concatenate all features
    X = np.hstack([X_structured, X_text])  # (n_samples, n_features_total)

    # Naive Bayes prediction
    # feature_log_prob has shape (n_classes, n_features)
    # class_log_prior has shape (n_classes,)
    #
    # For each sample x:
    #   log P(y=k | x) ∝ log P(y=k) + sum_j x_j * log P(x_j | y=k)
    log_joint = class_log_prior[np.newaxis, :] + X @ feature_log_prob.T  # (n_samples, n_classes)

    pred_indices = np.argmax(log_joint, axis=1)
    pred_labels = labels[pred_indices]

    return list(pred_labels)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 nb_pred.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    if len(sys.argv) == 3:
        output_csv = sys.argv[2]
    else:
        output_csv = "nb_predictions.csv"

    predictions = predict_all(input_csv)

    df_out = pd.DataFrame({"prediction": predictions})
    df_out.to_csv(output_csv, index=False)
    print(f"\nWrote {len(predictions)} predictions to {output_csv}")