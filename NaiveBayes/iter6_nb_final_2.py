"""
Train a Multinomial Naive Bayes model on all rows of training_data_clean.csv using:
    - 4 rating (Likert) features
    - 2 multi-select questions, each encoded using 8 target tasks -> 16 binary features total
    - 3 free-text questions combined into a bag-of-words with up to 3000 features

Model:
    - MultinomialNB with Laplace smoothing alpha = ALPHA

Outputs:
    This script saves everything needed for nb_pred.py (which cannot use sklearn):

    - nb_final_1_params.npz
        class_log_prior : shape (n_classes,)
        feature_log_prob: shape (n_classes, n_features)
        labels          : class labels in the order used by the model
        alpha           : smoothing parameter used during training

    - nb_final_1_vocab.json
        A dict mapping words to column indices in the text feature part of the feature vector.

    - nb_final_1_config.json
        A small JSON with:
            * rating column names
            * neutral rating value
            * multi-select column names
            * list of target task substrings
            * text column names
            * number of structured features (ratings + multi-select)
            * max_text_features (3000)
"""
import numpy as np
import pandas as pd
import re
import json
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

file_name = "../DataSet/training_data_clean.csv"

# All rating / Likert-style questions we want to use (4 columns)
RATING_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

# Multi-select columns
BEST_TASKS_COL = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
SUBOPT_TASKS_COL = 'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'

# Target tasks we will track for both multi-select questions
target_tasks = [
    'Math computations',
    'Writing or debugging code',
    'Data processing or analysis',
    'Explaining complex concepts simply',
    'Writing or editing essays/reports',
    'Drafting professional text',
    'Brainstorming or generating creative ideas',
    'Converting content between formats',
]

# Text columns to combine into one bag-of-words text field
TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

# Neutral rating value used when a rating is missing or unparsable
NEUTRAL_RATING_VALUE = 3

# Max number of text features (vocabulary size cap for CountVectorizer)
MAX_TEXT_FEATURES = 3000

# Hyperparameter to tune: Laplace smoothing alpha
# After running a tuning script, set ALPHA to the best value found (e.g. 0.5, 1.0, or 1.5).
ALPHA = 1.5


def process_multiselect(series, target_tasks):
    """
    Convert multiselect strings to lists, keeping only specified features.
    If a response is missing, we treat it as selecting nothing.
    """
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            text = str(response)
            present_tasks = [task for task in target_tasks if task in text]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing or unparsable responses.
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def build_rating_matrix(df, rating_cols, neutral_value=3):
    """
    Build a numeric feature matrix from multiple rating columns.

    For each column:
        - Apply extract_rating() to get numbers.
        - Replace missing values (None/NaN) with `neutral_value`.
        - Cast to int and reshape to a column vector.

    Returns
    -------
    X_ratings : np.ndarray of shape (n_samples, len(rating_cols))
    """
    columns = []
    for col in rating_cols:
        raw = df[col].apply(extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        columns.append(filled.to_numpy().reshape(-1, 1))
    X_ratings = np.hstack(columns)
    return X_ratings


def build_text_series(df, text_cols):
    """
    Combine several text columns into a single string per row.

    For each row:
        combined_text = col1 + " " + col2 + " " + col3

    Missing values are treated as empty strings.
    """
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined


def main():
    df = pd.read_csv(file_name)
    n_rows = df.shape[0]
    print(f"Loaded {n_rows} rows from {file_name}")

    # Structured features
    X_ratings = build_rating_matrix(df, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    best_tasks_lists = process_multiselect(df[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists = process_multiselect(df[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)            # (n_samples, 8)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(subopt_tasks_lists)  # (n_samples, 8)

    X_structured = np.hstack([
        X_ratings,
        best_tasks_encoded,
        suboptimal_tasks_encoded,
    ])

    print("Structured feature shape:", X_structured.shape)

    # Text features (bag-of-words counts)
    all_text = build_text_series(df, TEXT_COLS)

    vectorizer = CountVectorizer(
        max_features=MAX_TEXT_FEATURES,
        lowercase=True,
    )

    X_text_sparse = vectorizer.fit_transform(all_text)
    X_text = X_text_sparse.toarray()
    print("Text feature shape:", X_text.shape)

    # Concatenate structured + text
    X = np.hstack([X_structured, X_text])
    y = df['label'].values

    print("Final feature matrix shape:", X.shape)
    print("Distinct labels:", np.unique(y))

    # Train Multinomial NB
    nb = MultinomialNB(alpha=ALPHA)
    nb.fit(X, y)

    train_acc = nb.score(X, y)
    print(f"Training accuracy on all data (alpha={ALPHA}): {train_acc:.3f}")

    # Extract parameters
    class_log_prior = nb.class_log_prior_     # (n_classes,)
    feature_log_prob = nb.feature_log_prob_   # (n_classes, n_features)
    labels = nb.classes_                      # (n_classes,)

    print("class_log_prior shape:", class_log_prior.shape)
    print("feature_log_prob shape:", feature_log_prob.shape)
    print("Labels order:", labels)

    # Extract vocabulary from CountVectorizer
    vocab = vectorizer.vocabulary_            # dict: word -> column index
    vocab_json = {str(k): int(v) for k, v in vocab.items()}
    print(f"Vocabulary size: {len(vocab_json)}")

    # Save outputs
    out_dir = "."
    params_path = os.path.join(out_dir, "nb_final_2_params.npz")
    vocab_path = os.path.join(out_dir, "nb_final_2_vocab.json")
    config_path = os.path.join(out_dir, "nb_final_2_config.json")

    # Save numpy arrays and meta info in a compressed .npz file
    np.savez(
        params_path,
        class_log_prior=class_log_prior,
        feature_log_prob=feature_log_prob,
        labels=labels,
        alpha=np.array([ALPHA], dtype=float),
    )
    print(f"Saved NB model parameters to: {params_path}")

    # Save vocabulary as JSON
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)
    print(f"Saved vectorizer vocabulary to: {vocab_path}")

    # Save configuration info as JSON
    config = {
        "rating_cols": RATING_COLS,
        "neutral_rating_value": NEUTRAL_RATING_VALUE,
        "best_tasks_col": BEST_TASKS_COL,
        "subopt_tasks_col": SUBOPT_TASKS_COL,
        "target_tasks": target_tasks,
        "text_cols": TEXT_COLS,
        "n_structured_features": int(X_structured.shape[1]),
        "max_text_features": MAX_TEXT_FEATURES,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()