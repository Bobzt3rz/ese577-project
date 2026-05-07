import argparse
import json
import random
import re
from collections import Counter

import numpy as np


PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def load_sentences(path):
    english = []
    german = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng = preprocess(parts[0])
                ger = preprocess(parts[1])
                if eng and ger:
                    english.append(eng)
                    german.append(ger)

    return english, german


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def nearest_words(word, vocab, idx_to_word, normalized_embeddings, top_n):
    idx = vocab.get(word, UNK)
    sims = normalized_embeddings @ normalized_embeddings[idx]
    sims[idx] = -np.inf

    for special in SPECIAL_TOKENS:
        special_idx = vocab.get(special)
        if special_idx is not None:
            sims[special_idx] = -np.inf

    top_n = min(top_n, len(sims) - len(SPECIAL_TOKENS) - 1)
    top_indices = np.argpartition(-sims, range(top_n))[:top_n]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [(idx_to_word[i], sims[i]) for i in top_indices]


def context_coherence(sentences, vocab, normalized_embeddings, rng, sample_size, window_size):
    known_sentences = [
        [vocab[token] for token in sentence if token in vocab]
        for sentence in sentences
    ]
    known_sentences = [sentence for sentence in known_sentences if len(sentence) >= 3]

    if not known_sentences:
        return None

    vocab_indices = [idx for word, idx in vocab.items() if word not in SPECIAL_TOKENS]
    observed = []
    random_baseline = []

    attempts = 0
    max_attempts = sample_size * 20
    while len(observed) < sample_size and attempts < max_attempts:
        attempts += 1
        sentence = rng.choice(known_sentences)
        center_pos = rng.randrange(len(sentence))
        center_idx = sentence[center_pos]

        left = sentence[max(0, center_pos - window_size):center_pos]
        right = sentence[center_pos + 1:center_pos + 1 + window_size]
        context = [idx for idx in left + right if idx not in (PAD, SOS, EOS, UNK)]
        if not context:
            continue

        random_context = rng.sample(vocab_indices, k=min(len(context), len(vocab_indices)))
        center = normalized_embeddings[center_idx]
        observed.append(float(np.mean(normalized_embeddings[context] @ center)))
        random_baseline.append(float(np.mean(normalized_embeddings[random_context] @ center)))

    if not observed:
        return None

    observed = np.array(observed)
    random_baseline = np.array(random_baseline)
    return {
        "observed": float(observed.mean()),
        "random": float(random_baseline.mean()),
        "margin": float((observed - random_baseline).mean()),
        "samples": len(observed),
    }


def coverage(sentences, vocab, max_len):
    total = 0
    unknown = 0
    kept_sentences = 0

    for sentence in sentences:
        if len(sentence) > max_len:
            continue
        kept_sentences += 1
        for token in sentence:
            total += 1
            if token not in vocab:
                unknown += 1

    unk_rate = unknown / total if total else 0.0
    return kept_sentences, total, unknown, unk_rate


def report_language(name, sentences, vocab, embeddings, probes, rng, args):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    normalized = normalize_rows(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    token_counts = Counter(token for sentence in sentences for token in sentence)

    print(f"\n== {name} ==")
    print(f"Vocab size:       {len(vocab):,}")
    print(f"Embedding shape:  {embeddings.shape}")
    print(f"Finite values:    {np.isfinite(embeddings).all()}")
    print(f"Mean norm:        {norms.mean():.4f}")
    print(f"Median norm:      {np.median(norms):.4f}")
    print(f"Min/max norm:     {norms.min():.4f} / {norms.max():.4f}")

    for special in SPECIAL_TOKENS:
        idx = vocab.get(special)
        if idx is not None:
            print(f"{special:>5} norm:       {norms[idx]:.4f}")

    kept, total, unknown, unk_rate = coverage(sentences, vocab, args.max_len)
    print(f"Seq2seq coverage: {kept:,} sentences <= {args.max_len} tokens")
    print(f"UNK rate:         {unknown:,}/{total:,} ({unk_rate:.4%})")

    coherence = context_coherence(
        sentences,
        vocab,
        normalized,
        rng,
        args.samples,
        args.window_size,
    )
    if coherence:
        print(
            "Context cosine:  "
            f"observed {coherence['observed']:.4f}, "
            f"random {coherence['random']:.4f}, "
            f"margin {coherence['margin']:.4f} "
            f"({coherence['samples']:,} samples)"
        )

    print("\nNearest words:")
    for word in probes:
        if word not in vocab:
            print(f"  {word!r}: not in vocab")
            continue
        count = token_counts.get(word, 0)
        neighbors = nearest_words(word, vocab, idx_to_word, normalized, args.top_n)
        formatted = ", ".join(f"{neighbor} ({score:.2f})" for neighbor, score in neighbors)
        print(f"  {word!r} freq={count:,}: {formatted}")


def main():
    parser = argparse.ArgumentParser(description="Validate saved CBOW embedding artifacts.")
    parser.add_argument("--data", default="deu.txt")
    parser.add_argument("--english-vocab", default="english_vocab.json")
    parser.add_argument("--german-vocab", default="german_vocab.json")
    parser.add_argument("--english-embeddings", default="english_embeddings.npy")
    parser.add_argument("--german-embeddings", default="german_embeddings.npy")
    parser.add_argument("--max-len", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    english_sentences, german_sentences = load_sentences(args.data)
    english_vocab = load_vocab(args.english_vocab)
    german_vocab = load_vocab(args.german_vocab)
    english_embeddings = np.load(args.english_embeddings)
    german_embeddings = np.load(args.german_embeddings)

    if english_embeddings.shape[0] != len(english_vocab):
        raise ValueError("English embedding rows do not match English vocab size.")
    if german_embeddings.shape[0] != len(german_vocab):
        raise ValueError("German embedding rows do not match German vocab size.")
    if english_embeddings.shape[1] != german_embeddings.shape[1]:
        raise ValueError("English and German embedding dimensions differ.")

    print(f"Loaded {len(english_sentences):,} sentence pairs")
    print(f"Shared embedding dim: {english_embeddings.shape[1]}")
    print(
        "\nNote: this script validates saved embedding quality and artifact consistency. "
        "True held-out CBOW loss would require saving the trained CBOW classifier head."
    )

    report_language(
        "English",
        english_sentences,
        english_vocab,
        english_embeddings,
        ["good", "bad", "house", "car", "go", "went", "eat", "read", "station", "student"],
        rng,
        args,
    )
    report_language(
        "German",
        german_sentences,
        german_vocab,
        german_embeddings,
        ["gut", "schlecht", "haus", "auto", "gehen", "gegangen", "essen", "lesen", "bahnhof", "student"],
        rng,
        args,
    )


if __name__ == "__main__":
    main()
