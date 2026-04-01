# ==============================================================================
# CS481 - Spring 2026 - Programming Assignment 01
# Part D: Model Comparison using Perplexity
# Student: Ayaan Khan
# A-Number: A20505209
#
# Description: Compares two bigram language models using perplexity:
#   Model C - Stopwords removed, MLE (no smoothing)
#   Model D - Stopwords kept, Laplace (add-1) smoothing
#
# Modification rationale: Removing stopwords breaks natural bigram chains
# (e.g., "this is good" becomes just "good" after removing "this" and "is").
# Model D keeps stopwords to preserve natural word adjacency and applies
# Laplace smoothing to handle unseen bigrams (no zero probabilities).
# ==============================================================================

import nltk
import math
from nltk.corpus import brown, stopwords

nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)

# MODEL C: Bigram model with stopword removal, no smoothing (replicates Part C)

def build_model_c(stop_words):
    """
    Build Part C's model: lowercase, alphabetic only, stopwords removed, MLE bigrams.
    """
    bigram_list = []
    all_words = []

    for sentence in brown.sents():
        filtered = [w.lower() for w in sentence
                    if w.isalpha() and w.lower() not in stop_words]
        if len(filtered) < 2:
            continue
        all_words.extend(filtered)
        bigram_list.extend(list(nltk.bigrams(filtered)))

    cfd = nltk.ConditionalFreqDist(bigram_list)
    unigram_fd = nltk.FreqDist(all_words)
    vocab_size = len(unigram_fd)

    return cfd, unigram_fd, vocab_size


def prob_model_c(w1, w2, cfd, unigram_fd):
    """
    MLE bigram probability: P(w2|w1) = count(w1, w2) / count(w1).
    Returns 0 for unseen bigrams or unknown words.
    """
    count_w1 = unigram_fd[w1]
    if count_w1 == 0:
        return 0.0
    return cfd[w1][w2] / count_w1


# MODEL D: Bigram model WITH stopwords, Laplace (add-1) smoothing

def build_model_d():
    """
    Build the improved model: lowercase, alphabetic only, stopwords KEPT,
    Laplace smoothing applied during probability calculation.
    """
    bigram_list = []
    all_words = []

    for sentence in brown.sents():
        filtered = [w.lower() for w in sentence if w.isalpha()]
        if len(filtered) < 2:
            continue
        all_words.extend(filtered)
        bigram_list.extend(list(nltk.bigrams(filtered)))

    cfd = nltk.ConditionalFreqDist(bigram_list)
    unigram_fd = nltk.FreqDist(all_words)
    vocab_size = len(unigram_fd)

    return cfd, unigram_fd, vocab_size


def prob_model_d(w1, w2, cfd, unigram_fd, vocab_size):
    """
    Laplace-smoothed bigram probability:
      P(w2|w1) = (count(w1, w2) + 1) / (count(w1) + V)
    where V = vocabulary size.
    This ensures no bigram ever has zero probability.
    """
    count_w1 = unigram_fd[w1]
    count_bigram = cfd[w1][w2]
    return (count_bigram + 1) / (count_w1 + vocab_size)


# PERPLEXITY CALCULATION

def compute_perplexity_c(sentence_str, cfd, unigram_fd):
    """
    Compute perplexity for Model C.

    Formula:
      PP(S) = P(S) ^ (-1/N)
    where N = number of words, and
      P(S) = P(w1|<s>) * P(w2|w1) * ... * P(</s>|wn)

    Sentence boundary bigrams assumed P = 0.25.
    If any bigram has P=0, perplexity is infinite.

    Uses log-space to avoid underflow:
      log PP(S) = -1/N * sum(log P(each bigram))
    """
    words = sentence_str.lower().split()
    N = len(words)
    if N == 0:
        return float('inf'), 0.0, []

    log_prob_sum = 0.0
    bigram_details = []

    # Sentence start boundary
    p_start = 0.25
    bigram_details.append(("<s>", words[0], p_start))
    log_prob_sum += math.log(p_start)

    # Internal bigrams
    for i in range(len(words) - 1):
        p = prob_model_c(words[i], words[i + 1], cfd, unigram_fd)
        bigram_details.append((words[i], words[i + 1], p))
        if p == 0:
            return float('inf'), 0.0, bigram_details
        log_prob_sum += math.log(p)

    # Sentence end boundary
    p_end = 0.25
    bigram_details.append((words[-1], "</s>", p_end))
    log_prob_sum += math.log(p_end)

    prob_s = math.exp(log_prob_sum)
    perplexity = math.exp(-log_prob_sum / N)

    return perplexity, prob_s, bigram_details


def compute_perplexity_d(sentence_str, cfd, unigram_fd, vocab_size):
    """
    Compute perplexity for Model D (Laplace smoothing, stopwords kept).
    Same formula as Model C but uses Laplace-smoothed probabilities.
    Perplexity is always finite due to smoothing.
    """
    words = sentence_str.lower().split()
    N = len(words)
    if N == 0:
        return float('inf'), 0.0, []

    log_prob_sum = 0.0
    bigram_details = []

    # Sentence start boundary
    p_start = 0.25
    bigram_details.append(("<s>", words[0], p_start))
    log_prob_sum += math.log(p_start)

    # Internal bigrams
    for i in range(len(words) - 1):
        p = prob_model_d(words[i], words[i + 1], cfd, unigram_fd, vocab_size)
        bigram_details.append((words[i], words[i + 1], p))
        log_prob_sum += math.log(p)

    # Sentence end boundary
    p_end = 0.25
    bigram_details.append((words[-1], "</s>", p_end))
    log_prob_sum += math.log(p_end)

    prob_s = math.exp(log_prob_sum)
    perplexity = math.exp(-log_prob_sum / N)

    return perplexity, prob_s, bigram_details


# DISPLAY

def display_results(sentence, pp_c, prob_c, details_c, pp_d, prob_d, details_d):
    """Display detailed perplexity comparison for one sentence."""
    print(f"\n{'='*60}")
    print(f"Sentence: \"{sentence}\"")
    print(f"{'='*60}")

    # Model C
    print(f"\n--- Model C (Stopwords removed, No smoothing) ---")
    for w1, w2, p in details_c:
        print(f"  P({w2}|{w1}) = {p:.10f}")
    print(f"  P(S) = {prob_c:.15e}")
    if pp_c == float('inf'):
        print(f"  Perplexity = INFINITY (zero-probability bigram)")
    else:
        print(f"  Perplexity = {pp_c:.4f}")

    # Model D
    print(f"\n--- Model D (Stopwords kept, Laplace smoothing) ---")
    for w1, w2, p in details_d:
        print(f"  P({w2}|{w1}) = {p:.10f}")
    print(f"  P(S) = {prob_d:.15e}")
    print(f"  Perplexity = {pp_d:.4f}")


def main():
    stop_words = set(stopwords.words('english'))

    # Build both models
    print("Building Model C (stopwords removed, MLE)...")
    cfd_c, uni_c, vocab_c = build_model_c(stop_words)
    print(f"  Vocabulary size: {vocab_c}")

    print("Building Model D (stopwords kept, Laplace smoothing)...")
    cfd_d, uni_d, vocab_d = build_model_d()
    print(f"  Vocabulary size: {vocab_d}")

    # Test sentences from the assignment
    test_sentences = [
        "this is good",
        "this is bad",
        "I am good",
        "I am bad",
        "good afternoon",
        "good day"
    ]

    # Print formulas
    print(f"\n{'='*60}")
    print("PERPLEXITY COMPARISON: Model C vs Model D")
    print(f"{'='*60}")
    print("\nFormulas:")
    print("  MLE (Model C):     P(w2|w1) = count(w1, w2) / count(w1)")
    print("  Laplace (Model D): P(w2|w1) = (count(w1, w2) + 1) / (count(w1) + V)")
    print("  P(S) = P(w1|<s>) * prod(P(wi|wi-1)) * P(</s>|wn)")
    print("  PP(S) = P(S) ^ (-1/N),  where N = number of words")
    print("  Sentence boundary bigrams: P = 0.25 (assumed)")

    # Compute and display perplexity for each test sentence
    for sentence in test_sentences:
        pp_c, prob_c, det_c = compute_perplexity_c(sentence, cfd_c, uni_c)
        pp_d, prob_d, det_d = compute_perplexity_d(sentence, cfd_d, uni_d, vocab_d)
        display_results(sentence, pp_c, prob_c, det_c, pp_d, prob_d, det_d)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Sentence':<20} {'PP (Model C)':<20} {'PP (Model D)':<20}")
    print("-" * 60)
    for sentence in test_sentences:
        pp_c, _, _ = compute_perplexity_c(sentence, cfd_c, uni_c)
        pp_d, _, _ = compute_perplexity_d(sentence, cfd_d, uni_d, vocab_d)
        pp_c_str = f"{pp_c:.4f}" if pp_c != float('inf') else "INFINITY"
        pp_d_str = f"{pp_d:.4f}"
        print(f"{sentence:<20} {pp_c_str:<20} {pp_d_str:<20}")

    # Conclusions
    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print(f"{'='*60}")
    print("""
Model C (stopword removal, no smoothing) produces INFINITE perplexity for most
test sentences because common function words like "is", "am", "this", and "I"
are stopwords and get removed from the training corpus. Bigrams containing
these words have zero probability, making P(S) = 0 and PP = infinity.

Model D (stopwords retained, Laplace smoothing) produces FINITE perplexity for
ALL sentences. Two key improvements contribute:

  1. Keeping stopwords preserves natural bigram chains that connect function
     words to content words (e.g., "this is", "is good", "I am").

  2. Laplace smoothing ensures every bigram has non-zero probability by adding
     a count of 1 to all bigrams: P(w2|w1) = (count + 1) / (count(w1) + V).
     This handles unseen bigrams gracefully.

Model D is clearly superior for these test sentences. Lower perplexity
indicates a better model — the model assigns higher probability to natural
sentences. The test sentences are simple, everyday phrases that rely heavily
on function words (stopwords), which Model C cannot model at all.

This demonstrates that aggressive stopword removal, while useful for tasks
like information retrieval, can be harmful for language modeling where
capturing the full sequential structure of language is essential.
""")


if __name__ == '__main__':
    main()
