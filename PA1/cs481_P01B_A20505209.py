
import nltk
from nltk.corpus import brown

# Download required NLTK data
nltk.download('brown', quiet=True)


def build_bigram_model():
    """
    Build a bigram model from the Brown corpus.

    Uses brown.sents() to respect sentence boundaries. All words are lowercased.
    No stopword removal for Part B.

    Returns:
        cfd: ConditionalFreqDist where cfd[w1][w2] = count of bigram (w1, w2)
        unigram_fd: FreqDist where unigram_fd[w] = count of word w
    """
    bigram_list = []
    all_words = []

    for sentence in brown.sents():
        words = [w.lower() for w in sentence]
        all_words.extend(words)
        # Build bigrams within each sentence only (no cross-sentence bigrams)
        bigram_list.extend(list(nltk.bigrams(words)))

    cfd = nltk.ConditionalFreqDist(bigram_list)
    unigram_fd = nltk.FreqDist(all_words)

    return cfd, unigram_fd


def bigram_probability(w1, w2, cfd, unigram_fd):
    """
    Compute P(w2 | w1) using Maximum Likelihood Estimation.
    P(w2|w1) = count(w1, w2) / count(w1)
    Returns 0.0 if w1 was never observed.
    """
    count_w1 = unigram_fd[w1]
    if count_w1 == 0:
        return 0.0
    return cfd[w1][w2] / count_w1


def compute_sentence_probability(sentence_str, cfd, unigram_fd):
    """
    Compute P(S) for a sentence using the bigram model.

    P(S) = P(w1|<s>) * P(w2|w1) * ... * P(wn|wn-1) * P(</s>|wn)

    Sentence boundary assumptions:
      - P(first_word | <s>) = 0.25
      - P(</s> | last_word) = 0.25

    Returns:
        probability: P(S) as a float
        bigram_details: list of (w1, w2, prob) tuples for display
    """
    words = sentence_str.lower().split()

    if len(words) == 0:
        return 0.0, []

    bigram_details = []
    probability = 1.0

    # Sentence start -> first word
    p_start = 0.25
    bigram_details.append(("<s>", words[0], p_start))
    probability *= p_start

    # Internal bigrams
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        p = bigram_probability(w1, w2, cfd, unigram_fd)
        bigram_details.append((w1, w2, p))
        probability *= p

    # Last word -> sentence end
    p_end = 0.25
    bigram_details.append((words[-1], "</s>", p_end))
    probability *= p_end

    return probability, bigram_details


def main():
    print("Building bigram language model from Brown corpus...")
    cfd, unigram_fd = build_bigram_model()
    print("Model built successfully.\n")

    # Get sentence from user
    sentence = input("Enter a sentence: ")

    probability, bigram_details = compute_sentence_probability(sentence, cfd, unigram_fd)

    # Display results
    print(f"\nSentence (lowercased): \"{sentence.lower()}\"")
    print(f"\n{'Bigram':<35} {'P(w2|w1)':<15}")
    print("-" * 50)

    for w1, w2, p in bigram_details:
        bigram_str = f"({w1}, {w2})"
        print(f"{bigram_str:<35} {p:.10f}")

    print(f"\nP(S) = {probability:.15e}")

    if probability == 0:
        print("Note: P(S) = 0 because at least one bigram was never observed in the corpus.")


if __name__ == '__main__':
    main()
