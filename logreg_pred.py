import os
import json
import re
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_PARAMS_PATH = os.path.join(_BASE_DIR, "logreg_final_1_params.npz")
_VOCAB_PATH = os.path.join(_BASE_DIR, "logreg_final_1_vocab.json")
_CONFIG_PATH = os.path.join(_BASE_DIR, "logreg_final_1_config.json")

def _load_model_and_config():
    """
    Load model parameters (W, b, labels), vocabulary, and configuration.

    W : np.ndarray of shape (n_classes, n_features). Logistic regression weights.
    b : np.ndarray of shape (n_classes,). Logistic regression biases.
    labels : np.ndarray of shape (n_classes,) Class labels in the order used by the model.

    vocab : dict[str, int]. Mapping from token -> column index in the text feature block.

    config : dict
        Dictionary containing:
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
    W = params["W"]
    b = params["b"]
    labels = params["labels"]

    with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return W, b, labels, vocab, config

_RATING_REGEX = re.compile(r"^(\d+)") # capture leading integer
_TOKEN_PATTERN = re.compile(r"\b\w\w+\b") # mimic CountVectorizer's default token_pattern

def _extract_rating(response):
    match = _RATING_REGEX.match(str(response))
    return int(match.group(1)) if match else None

def _build_rating_matrix(df, rating_cols, neutral_value):
    columns = []
    for col in rating_cols:
        raw = df[col].apply(_extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        columns.append(filled.to_numpy().reshape(-1, 1))
    X_ratings = np.hstack(columns)
    return X_ratings

def _process_multiselect(series, target_tasks):
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
    Convert lists of tasks into a binary matrix using a FIXED task order
    We mimic sklearn's MultiLabelBinarizer with default behavior: columns are in sorted label order

    Parameters
        processed_lists : list[list[str]] 
            Output of _process_multiselect (list of tasks per row).
        target_tasks_sorted : list[str]
            Sorted list of all tasks used during training.
    
    Returns
        X_multi : np.ndarray of shape (n_samples, len(target_tasks_sorted))
    """
    n_samples = len(processed_lists)
    n_tasks = len(target_tasks_sorted)
    X_multi = np.zeros((n_samples, n_tasks), dtype=np.int64)

    # Map task -> column index according to sorted order
    task_to_idx = {task: i for i, task in enumerate(target_tasks_sorted)}

    for i, task_list in enumerate(processed_lists):
        for task in task_list:
            j = task_to_idx.get(task)
            if j is not None:
                X_multi[i, j] = 1
    return X_multi

def _build_text_series(df, text_cols):
    """
    Combine multiple text columns into a single string per row.
    """
    combined = df[text_cols[0]].fillna("")
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna("")
    return combined

def _build_bow_matrix(text_series, vocab, max_text_features):
    """
    Build a bag-of-words matrix using the saved vocabulary.

    Parameter:
        text_series : pd.Series of str
        vocab : dict[str, int]
            Mapping from token -> index (0 <= index < max_text_features).
        max_text_features : int

    Return:
        X_text : np.ndarray of shape (n_samples, max_text_features)
    """
    n_samples = text_series.shape[0]
    X_text = np.zeros((n_samples, max_text_features), dtype=np.float32)

    # vocab keys are tokens, values are integer column indices
    for i, text in enumerate(text_series):
        if pd.isna(text):
            text = ""
        text = str(text).lower()
        for match in _TOKEN_PATTERN.finditer(text):
            token = match.group(0)
            idx = vocab.get(token)
            if idx is not None and 0 <= idx < max_text_features:
                X_text[i, idx] += 1.0 # count occurrences
    return X_text

def predict_all(csv_path):
    W, b, labels, vocab, config = _load_model_and_config()

    rating_cols = config["rating_cols"]
    neutral_value = config["neutral_rating_value"]
    best_col = config["best_tasks_col"]
    subopt_col = config["subopt_tasks_col"]
    target_tasks = config["target_tasks"]
    text_cols = config["text_cols"]
    n_structured = int(config["n_structured_features"])
    max_text_features = int(config["max_text_features"])

    # From training MultiLabelBinarizer used sorted label order.
    target_tasks_sorted = sorted(target_tasks)

    df = pd.read_csv(csv_path)

    X_ratings = _build_rating_matrix(df, rating_cols, neutral_value)

    best_lists = _process_multiselect(df[best_col], target_tasks)
    subopt_lists = _process_multiselect(df[subopt_col], target_tasks)

    X_best = _multiselect_to_binary(best_lists, target_tasks_sorted)
    X_subopt = _multiselect_to_binary(subopt_lists, target_tasks_sorted)

    X_structured = np.hstack([X_ratings, X_best, X_subopt])

    text_series = _build_text_series(df, text_cols)
    X_text = _build_bow_matrix(text_series, vocab, max_text_features)

    # Concatenate structured + text to get the full feature matrix.
    X = np.hstack([X_structured, X_text])

    # Apply the linear model: scores = X @ W^T + b
    #    X: (n_samples, n_features)
    #    W: (n_classes, n_features)
    #    -> scores: (n_samples, n_classes)
    scores = X.dot(W.T) + b
    
    # Take argmax over classes to get the predicted class index, then map back to the label strings.
    pred_indices = np.argmax(scores, axis=1)
    pred_labels = labels[pred_indices]

    return list(pred_labels)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 logreg_pred.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    if len(sys.argv) == 3:
        output_csv = sys.argv[2]
    else:
        output_csv = "predictions.csv"
    
    predictions = predict_all(input_csv)

    df_out = pd.DataFrame({"prediction": predictions})
    df_out.to_csv(output_csv, index=False)
    print(f"\nWrote {len(predictions)} predictions to {output_csv}")