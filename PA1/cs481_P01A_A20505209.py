import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import brown, reuters, stopwords

nltk.download('brown', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)


def get_filtered_words(corpus, stop_words):
    """
    Extract words from a corpus, lowercase them, keep only alphabetic tokens,
    and remove stop words.
    """
    return [w.lower() for w in corpus.words()
            if w.isalpha() and w.lower() not in stop_words]


def display_top_n(freq_dist, corpus_name, n=10):
    """Print the top N most common words and their frequencies."""
    print(f"\nTop {n} words in {corpus_name} (stopwords removed):")
    print(f"{'Rank':<6} {'Word':<20} {'Frequency':<10}")
    print("-" * 36)
    for rank, (word, freq) in enumerate(freq_dist.most_common(n), 1):
        print(f"{rank:<6} {word:<20} {freq:<10}")


def plot_zipf(freq_dist, corpus_name, filename):
    """
    Generate a log10(rank) vs log10(frequency) plot for ranks 1-1000.
    Saves the plot to the given filename.
    """
    most_common = freq_dist.most_common(1000)
    ranks = np.arange(1, len(most_common) + 1)
    frequencies = np.array([freq for _, freq in most_common])

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)

    plt.figure(figsize=(10, 6))
    plt.plot(log_ranks, log_freqs, 'b-', linewidth=1)
    plt.xlabel('log10(Rank)')
    plt.ylabel('log10(Frequency)')
    plt.title(f"Zipf's Law: {corpus_name} Corpus (Ranks 1-1000, Stopwords Removed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")


def compute_unigram_probability(freq_dist, word):
    """
    Compute unigram probability: P(word) = count(word) / total_words.
    Returns (count, total, probability).
    """
    count = freq_dist[word]
    total = freq_dist.N()  # total number of word tokens
    prob = count / total if total > 0 else 0
    return count, total, prob


def main():
    stop_words = set(stopwords.words('english'))

    # Process both corpora
    print("Processing Brown corpus...")
    brown_words = get_filtered_words(brown, stop_words)
    brown_fd = nltk.FreqDist(brown_words)

    print("Processing Reuters corpus...")
    reuters_words = get_filtered_words(reuters, stop_words)
    reuters_fd = nltk.FreqDist(reuters_words)

    # Display top 10 words for both corpora
    display_top_n(brown_fd, "Brown")
    display_top_n(reuters_fd, "Reuters")

    # Generate Zipf's law plots
    print("\nGenerating Zipf's law plots...")
    plot_zipf(brown_fd, "Brown", "zipf_brown.png")
    plot_zipf(reuters_fd, "Reuters", "zipf_reuters.png")

    # Unigram probability analysis
    technical_word = "equilibrium"  # technical/seldom used word
    casual_word = "coffee"        # casual/daily-use word

    print(f"\n{'='*50}")
    print("Unigram Probability Analysis")
    print(f"{'='*50}")

    for corpus_name, fd in [("Brown", brown_fd), ("Reuters", reuters_fd)]:
        for word in [technical_word, casual_word]:
            count, total, prob = compute_unigram_probability(fd, word)
            print(f"\n{corpus_name} - '{word}':")
            print(f"  Count: {count}")
            print(f"  Total words (after filtering): {total}")
            print(f"  P('{word}') = {count}/{total} = {prob:.10f}")


if __name__ == '__main__':
    main()
