import re
from collections import Counter

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.split()

english_sentences = []
german_sentences  = []

with open("deu.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            eng = preprocess(parts[0])
            ger = preprocess(parts[1])
            # skip empty sentences after preprocessing
            if eng and ger:
                english_sentences.append(eng)
                german_sentences.append(ger)

# -- Sentence counts
print(f"Total sentence pairs: {len(english_sentences):,}")

# -- Vocabulary
english_vocab = Counter(w for s in english_sentences for w in s)
german_vocab  = Counter(w for s in german_sentences  for w in s)

print(f"\nEnglish:")
print(f"  Total tokens (word occurrences) : {sum(english_vocab.values()):,}")
print(f"  Unique words (vocab size)       : {len(english_vocab):,}")

print(f"\nGerman:")
print(f"  Total tokens (word occurrences) : {sum(german_vocab.values()):,}")
print(f"  Unique words (vocab size)       : {len(german_vocab):,}")

# -- Sentence length stats
eng_lens = [len(s) for s in english_sentences]
ger_lens  = [len(s) for s in german_sentences]

print(f"\nSentence length (in words):")
print(f"  {'':10s}  {'English':>10s}  {'German':>10s}")
print(f"  {'Average':10s}  {sum(eng_lens)/len(eng_lens):>10.1f}  {sum(ger_lens)/len(ger_lens):>10.1f}")
print(f"  {'Min':10s}  {min(eng_lens):>10d}  {min(ger_lens):>10d}")
print(f"  {'Max':10s}  {max(eng_lens):>10d}  {max(ger_lens):>10d}")

# -- Most common words
print(f"\nTop 10 most common English words:")
for word, count in english_vocab.most_common(10):
    print(f"  {word:15s} {count:,}")

print(f"\nTop 10 most common German words:")
for word, count in german_vocab.most_common(10):
    print(f"  {word:15s} {count:,}")

# -- Vocabulary coverage
# how many unique words cover 90% of all token occurrences
def coverage_vocab_size(vocab_counter, threshold=0.9):
    total = sum(vocab_counter.values())
    cumulative = 0
    for i, (word, count) in enumerate(vocab_counter.most_common()):
        cumulative += count
        if cumulative / total >= threshold:
            return i + 1

eng_cov = coverage_vocab_size(english_vocab)
ger_cov = coverage_vocab_size(german_vocab)
print(f"\nVocabulary needed to cover 90% of tokens:")
print(f"  English : {eng_cov:,} words")
print(f"  German  : {ger_cov:,} words")