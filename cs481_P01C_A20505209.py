
import nltk
from nltk.corpus import brown, stopwords

# Download required NLTK data
nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)


def build_bigram_model(stop_words):
    """
    Build a bigram model from the Brown corpus with stopwords removed.

    Process:
      1. Iterate over brown.sents()
      2. Lowercase, keep only alphabetic tokens, remove stopwords
      3. Build bigrams from the filtered tokens in each sentence
      4. Return ConditionalFreqDist and unigram FreqDist

    Bigrams are formed from the filtered sequence, so they connect
    non-stopword tokens that were adjacent after filtering.
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

    return cfd, unigram_fd


def get_top_next_words(word, cfd, unigram_fd, n=3):
    """
    Get the top N most likely words to follow the given word,
    along with their bigram probabilities P(next|word).

    Returns a list of (next_word, probability) tuples sorted by probability.
    """
    if word not in cfd:
        return []

    total = unigram_fd[word]  # count of w1, used as MLE denominator
    if total == 0:
        return []

    # Get the most common next words from the conditional frequency distribution
    top_words = cfd[word].most_common(n)
    return [(w, count / total) for w, count in top_words]


def main():
    stop_words = set(stopwords.words('english'))

    print("Building bigram model from Brown corpus (stopwords removed)...")
    cfd, unigram_fd = build_bigram_model(stop_words)
    print(f"Model built. Vocabulary size: {len(unigram_fd)}")

    generated_sentence = []

    # Get initial word from user
    while True:
        w1 = input("\nEnter the initial word: ").strip().lower()

        if unigram_fd[w1] == 0:
            print(f"'{w1}' was NOT found in the corpus.")
            print("Options:")
            print("  1. Try another word")
            print("  2. QUIT")
            choice = input("Your choice (1/2): ").strip()
            if choice == '2':
                print("Goodbye!")
                return
            continue

        generated_sentence.append(w1)
        break

    # Interactive generation loop
    current_word = w1
    while True:
        top_words = get_top_next_words(current_word, cfd, unigram_fd, n=3)

        if not top_words:
            print(f"\nNo bigrams found starting with '{current_word}'. Ending generation.")
            break

        # Display current sentence and menu
        print(f"\n{' '.join(generated_sentence)} ...")
        print(f"\nWhich word should follow:")
        for i, (word, prob) in enumerate(top_words, 1):
            bigram_str = f"{current_word} {word}"
            print(f"  {i}. {word}  P({bigram_str}) = {prob:.6f}")
        print(f"  4. QUIT")

        choice = input("\nYour choice: ").strip()

        if choice == '4':
            break
        elif choice in ('1', '2', '3'):
            idx = int(choice) - 1
            if idx < len(top_words):
                chosen_word = top_words[idx][0]
            else:
                # Fewer than 3 options available, default to first
                chosen_word = top_words[0][0]
        else:
            # Invalid input defaults to choice 1
            print("Invalid input. Defaulting to choice 1.")
            chosen_word = top_words[0][0]

        generated_sentence.append(chosen_word)
        current_word = chosen_word

    # Display final sentence
    print(f"\n{'='*50}")
    print(f"Generated sentence: {' '.join(generated_sentence)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
