"""
CS 481 Spring 2026 - Programming Assignment #02
Naive Bayes and K Nearest Neighbors Classifiers (from scratch)
Stock Market Sentiment Classification

Khan, Ayaan, A20505209
"""

import sys
import csv
import math
import re
import os


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse ALGO and TRAIN_SIZE from command line with defaults."""
    algo = 0
    train_size = 80

    args = sys.argv[1:]
    if len(args) != 2:
        return algo, train_size

    try:
        algo = int(args[0])
        if algo not in (0, 1):
            algo = 0
    except ValueError:
        algo = 0

    try:
        train_size = int(args[1])
        if train_size < 50 or train_size > 90:
            train_size = 80
    except ValueError:
        train_size = 80

    return algo, train_size


# ---------------------------------------------------------------------------
# 2. Data loading and preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase, remove punctuation/special chars, strip whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """Split cleaned text into word tokens."""
    return text.split()


def load_data(filepath):
    """Load CSV and return list of (tokens, label) tuples."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            try:
                label = int(row[1].strip())
            except ValueError:
                continue
            tokens = tokenize(clean_text(text))
            if tokens:
                data.append((tokens, label))
    return data


# ---------------------------------------------------------------------------
# 3. Dataset splitting and vocabulary
# ---------------------------------------------------------------------------

def split_data(data, train_pct):
    """Split into training (first train_pct%) and test (last 20%) sets."""
    n = len(data)
    train_end = int(n * train_pct / 100)
    test_start = int(n * 0.80)
    train_set = data[:train_end]
    test_set = data[test_start:]
    return train_set, test_set


def build_vocabulary(data):
    """Build vocabulary from ALL data (entire dataset)."""
    vocab = set()
    for tokens, _ in data:
        vocab.update(tokens)
    return vocab


def doc_to_bow(tokens):
    """Convert token list to word-count dictionary (non-binary BoW)."""
    bow = {}
    for w in tokens:
        bow[w] = bow.get(w, 0) + 1
    return bow


# ---------------------------------------------------------------------------
# 4. Naive Bayes
# ---------------------------------------------------------------------------

def train_naive_bayes(train_set, vocab):
    """
    Train Naive Bayes with add-1 smoothing.
    Returns priors and likelihoods dictionaries.
    """
    class_counts = {}
    word_counts = {}   # word_counts[label][word] = count
    total_words = {}   # total_words[label] = total word count in class

    for tokens, label in train_set:
        class_counts[label] = class_counts.get(label, 0) + 1
        if label not in word_counts:
            word_counts[label] = {}
            total_words[label] = 0
        for w in tokens:
            word_counts[label][w] = word_counts[label].get(w, 0) + 1
            total_words[label] += 1

    n_docs = len(train_set)
    vocab_size = len(vocab)

    # Priors: P(class)
    priors = {}
    for label in class_counts:
        priors[label] = class_counts[label] / n_docs

    # Likelihoods: P(word|class) with add-1 smoothing
    likelihoods = {}
    for label in class_counts:
        likelihoods[label] = {}
        denom = total_words[label] + vocab_size
        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (count + 1) / denom

    model = {
        'priors': priors,
        'likelihoods': likelihoods,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'total_words': total_words,
        'word_counts': word_counts,
    }
    return model


def predict_naive_bayes(tokens, model):
    """
    Predict class using log-space Naive Bayes.
    Returns (predicted_label, {label: probability}).
    """
    priors = model['priors']
    likelihoods = model['likelihoods']
    vocab = model['vocab']
    vocab_size = model['vocab_size']
    total_words = model['total_words']
    word_counts = model['word_counts']

    log_scores = {}
    for label in priors:
        log_scores[label] = math.log(priors[label])
        for w in tokens:
            if w in vocab:
                log_scores[label] += math.log(likelihoods[label][w])
            else:
                # Word not in vocab — use smoothing only
                denom = total_words[label] + vocab_size
                log_scores[label] += math.log(1 / denom)

    # Convert from log-space to linear probabilities (normalized)
    max_log = max(log_scores.values())
    exp_scores = {}
    for label in log_scores:
        exp_scores[label] = math.exp(log_scores[label] - max_log)

    total = sum(exp_scores.values())
    probs = {}
    for label in exp_scores:
        probs[label] = exp_scores[label] / total

    predicted = max(probs, key=probs.get)
    return predicted, probs


# ---------------------------------------------------------------------------
# 5. KNN
# ---------------------------------------------------------------------------

def cosine_similarity(bow1, bow2):
    """Compute cosine similarity between two BoW dictionaries."""
    # Dot product
    dot = 0
    for w in bow1:
        if w in bow2:
            dot += bow1[w] * bow2[w]

    if dot == 0:
        return 0.0

    # Magnitudes
    mag1 = math.sqrt(sum(v * v for v in bow1.values()))
    mag2 = math.sqrt(sum(v * v for v in bow2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def train_knn(train_set):
    """Store training data as BoW vectors for KNN (lazy learner)."""
    training_vectors = []
    for tokens, label in train_set:
        bow = doc_to_bow(tokens)
        training_vectors.append((bow, label))
    return training_vectors


def predict_knn(tokens, training_vectors, k=5):
    """Predict class using KNN with cosine similarity."""
    query_bow = doc_to_bow(tokens)

    # Compute similarities to all training documents
    similarities = []
    for bow, label in training_vectors:
        sim = cosine_similarity(query_bow, bow)
        similarities.append((sim, label))

    # Sort by similarity descending, take top k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k = similarities[:k]

    # Majority vote
    votes = {}
    for _, label in top_k:
        votes[label] = votes.get(label, 0) + 1

    predicted = max(votes, key=votes.get)
    return predicted


# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(predictions, actuals, positive_label=1):
    """Compute confusion matrix and all metrics."""
    tp = tn = fp = fn = 0
    for pred, actual in zip(predictions, actuals):
        if actual == positive_label and pred == positive_label:
            tp += 1
        elif actual != positive_label and pred != positive_label:
            tn += 1
        elif actual != positive_label and pred == positive_label:
            fp += 1
        else:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    fscore = (2 * precision * sensitivity / (precision + sensitivity)
              if (precision + sensitivity) > 0 else 0.0)

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'fscore': fscore,
    }


def display_metrics(metrics):
    """Print metrics in the required format."""
    print(f"  Number of true positives: {metrics['tp']}")
    print(f"  Number of true negatives: {metrics['tn']}")
    print(f"  Number of false positives: {metrics['fp']}")
    print(f"  Number of false negatives: {metrics['fn']}")
    print(f"  Sensitivity (recall): {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Negative predictive value: {metrics['npv']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F-score: {metrics['fscore']:.4f}")


# ---------------------------------------------------------------------------
# 7. Label helpers
# ---------------------------------------------------------------------------

LABEL_NAMES = {1: "Positive", -1: "Negative"}


def label_name(label):
    return LABEL_NAMES.get(label, str(label))


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    algo, train_size = parse_args()
    algo_name = "Naive Bayes" if algo == 0 else "K Nearest Neighbors"

    print("Khan, Ayaan, A20505209 solution:")
    print(f"Training set size: {train_size} %")
    print(f"Classifier type: {algo_name}")

    # Locate dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'stock_data.csv')
    if not os.path.exists(csv_path):
        csv_path = 'stock_data.csv'

    # Load data
    data = load_data(csv_path)
    vocab = build_vocabulary(data)
    train_set, test_set = split_data(data, train_size)

    # Train
    print("\nTraining classifier...")
    if algo == 0:
        model = train_naive_bayes(train_set, vocab)
    else:
        training_vectors = train_knn(train_set)

    # Test
    print("Testing classifier...\n")
    predictions = []
    actuals = []
    for tokens, label in test_set:
        if algo == 0:
            pred, _ = predict_naive_bayes(tokens, model)
        else:
            pred = predict_knn(tokens, training_vectors, k=5)
        predictions.append(pred)
        actuals.append(label)

    metrics = evaluate(predictions, actuals, positive_label=1)

    print("Test results / metrics:")
    display_metrics(metrics)

    # Interactive loop
    while True:
        print("\nEnter your sentence/document:")
        sentence = input("Sentence/document S: ")
        tokens = tokenize(clean_text(sentence))

        if algo == 0:
            pred, probs = predict_naive_bayes(tokens, model)
            print(f"  was classified as {label_name(pred)}.")
            print(f"  P(Positive | S) = {probs.get(1, 0.0):.4f}")
            print(f"  P(Negative | S) = {probs.get(-1, 0.0):.4f}")
        else:
            pred = predict_knn(tokens, training_vectors, k=5)
            print(f"  was classified as {label_name(pred)}.")

        response = input("\nDo you want to enter another sentence [Y/N]? ")
        if response.strip().upper() != 'Y':
            break


if __name__ == '__main__':
    main()
