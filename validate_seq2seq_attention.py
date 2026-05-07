import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_IDS = {PAD, SOS, EOS, UNK}


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def encode(tokens, vocab, max_len):
    indices = [vocab.get(token, UNK) for token in tokens[:max_len]]
    indices = [SOS] + indices + [EOS]
    indices += [PAD] * (max_len + 2 - len(indices))
    return indices


def encoded_len(tokens, max_len):
    return min(len(tokens), max_len) + 2


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pairs(path, max_len):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng = preprocess(parts[0])
                ger = preprocess(parts[1])
                if eng and ger and len(eng) <= max_len and len(ger) <= max_len:
                    pairs.append((eng, ger))
    return pairs


def split_pairs(pairs, val_split, test_split, seed, pair_limit):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    if pair_limit is not None:
        pairs = pairs[:pair_limit]

    test_size = int(len(pairs) * test_split)
    val_size = int(len(pairs) * val_split)
    train_size = len(pairs) - val_size - test_size

    return (
        pairs[:train_size],
        pairs[train_size:train_size + val_size],
        pairs[train_size + val_size:],
    )


class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_vocab, ger_vocab, max_len):
        self.src = [torch.tensor(encode(eng, eng_vocab, max_len)) for eng, ger in pairs]
        self.tgt = [torch.tensor(encode(ger, ger_vocab, max_len)) for eng, ger in pairs]
        self.src_lens = [encoded_len(eng, max_len) for eng, ger in pairs]
        self.raw_pairs = pairs

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_lens[idx]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.embedding.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, src, lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.shape[1],
        )
        return outputs, h, c


class DotAttention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.embedding.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.dropout = nn.Dropout(dropout)
        self.attention = DotAttention()
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_token, h, c, encoder_outputs, src_mask):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1))).squeeze(1)
        context, weights = self.attention(h[-1], encoder_outputs, src_mask)
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        out, (h, c) = self.lstm(lstm_input, (h, c))
        out = out.squeeze(1)
        logits = self.linear(torch.cat([out, context], dim=1))
        return logits, h, c, weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab = self.decoder.linear.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab, device=self.device)

        encoder_outputs, h, c = self.encoder(src, src_lens)
        src_mask = src != PAD
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, h, c, _ = self.decoder(input_token, h, c, encoder_outputs, src_mask)
            outputs[:, t, :] = pred
            input_token = pred.argmax(dim=1)

        return outputs


def build_model(args, eng_vocab, ger_vocab, eng_embeddings, ger_embeddings, device):
    embedding_dim = eng_embeddings.shape[1]
    if embedding_dim != ger_embeddings.shape[1]:
        raise ValueError("English and German embeddings must have the same dimension.")

    encoder = Encoder(
        len(eng_vocab),
        embedding_dim,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        eng_embeddings,
    )
    decoder = AttentionDecoder(
        len(ger_vocab),
        embedding_dim,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        ger_embeddings,
    )
    return Seq2SeqAttention(encoder, decoder, device).to(device)


def decode_batch(model, src, src_lens, ger_idx2word, max_len, device):
    model.eval()
    batch_size = src.shape[0]
    encoder_outputs, h, c = model.encoder(src, src_lens)
    src_mask = src != PAD
    input_t = torch.full((batch_size,), SOS, dtype=torch.long, device=device)
    decoded = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        pred, h, c, _ = model.decoder(input_t, h, c, encoder_outputs, src_mask)
        top_idx = pred.argmax(dim=1)
        for i, idx in enumerate(top_idx):
            if finished[i]:
                continue
            word = ger_idx2word.get(idx.item(), "<UNK>")
            if word == "<EOS>":
                finished[i] = True
            elif word not in ("<PAD>", "<SOS>", "<UNK>"):
                decoded[i].append(word)
        input_t = top_idx
        if finished.all():
            break

    return decoded


def translate_with_attention(model, sentence, eng_vocab, ger_idx2word, max_len, device):
    model.eval()
    tokens = preprocess(sentence)
    src = torch.tensor(encode(tokens, eng_vocab, max_len)).unsqueeze(0).to(device)
    src_lens = torch.tensor([encoded_len(tokens, max_len)]).to(device)
    src_tokens = ["<SOS>"] + tokens[:max_len] + ["<EOS>"]

    with torch.no_grad():
        encoder_outputs, h, c = model.encoder(src, src_lens)
        src_mask = src != PAD
        input_t = torch.tensor([SOS], device=device)

        visible_translation = []
        target_labels = []
        attention_rows = []

        for _ in range(max_len + 1):
            pred, h, c, weights = model.decoder(input_t, h, c, encoder_outputs, src_mask)
            top_idx = pred.argmax(dim=1)
            word = ger_idx2word.get(top_idx.item(), "<UNK>")

            target_labels.append(word)
            attention_rows.append(weights.squeeze(0).detach().cpu().numpy()[:len(src_tokens)])

            if word == "<EOS>":
                break
            if word not in ("<PAD>", "<SOS>", "<UNK>"):
                visible_translation.append(word)
            input_t = top_idx

    attention = np.array(attention_rows) if attention_rows else np.empty((0, 0))
    return " ".join(visible_translation), src_tokens, target_labels, attention


def evaluate_loss(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt, src_lens in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            src_lens = src_lens.to(device, non_blocking=True)
            output = model(src, src_lens, tgt)
            output = output[:, 1:, :].reshape(-1, vocab_size)
            gold = tgt[:, 1:].reshape(-1)
            loss = criterion(output, gold)
            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_bleu(model, loader, ger_idx2word, max_len, device):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt, src_lens in loader:
            src = src.to(device, non_blocking=True)
            src_lens = src_lens.to(device, non_blocking=True)
            decoded = decode_batch(model, src, src_lens, ger_idx2word, max_len, device)

            for i in range(src.shape[0]):
                ref_indices = tgt[i].tolist()
                ref_words = [
                    ger_idx2word.get(idx, "<UNK>")
                    for idx in ref_indices
                    if idx not in SPECIAL_IDS
                ]
                references.append([ref_words])
                hypotheses.append(decoded[i])

    smoother = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, smoothing_function=smoother)


def save_attention_heatmap(sentence, model, eng_vocab, ger_idx2word, max_len, device, output_dir, prefix):
    translation, src_tokens, target_labels, attention = translate_with_attention(
        model,
        sentence,
        eng_vocab,
        ger_idx2word,
        max_len,
        device,
    )
    if attention.size == 0 or not target_labels:
        print(f"Skipping heatmap for {sentence!r}: no output tokens.")
        return

    print_attention_diagnostics(sentence, translation, src_tokens, target_labels, attention)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{prefix}.npy")
        np.save(path, attention)
        print(f"matplotlib is unavailable; saved raw attention weights to {path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{prefix}.png")
    content_image_path = os.path.join(output_dir, f"{prefix}_content_only.png")
    weights_path = os.path.join(output_dir, f"{prefix}.npy")
    content_weights_path = os.path.join(output_dir, f"{prefix}_content_only.npy")
    np.save(weights_path, attention)

    plt.figure(figsize=(max(6, len(src_tokens) * 0.8), max(4, len(target_labels) * 0.45)))
    plt.imshow(attention, aspect="auto", cmap="viridis")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    plt.yticks(range(len(target_labels)), target_labels)
    plt.xlabel("English source tokens")
    plt.ylabel("German predicted tokens")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()

    content_tokens, content_target_labels, content_attention = content_only_attention(
        src_tokens,
        target_labels,
        attention,
    )
    if content_attention.size:
        np.save(content_weights_path, content_attention)
        plt.figure(figsize=(max(6, len(content_tokens) * 0.8), max(4, len(content_target_labels) * 0.45)))
        plt.imshow(content_attention, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        plt.xticks(range(len(content_tokens)), content_tokens, rotation=45, ha="right")
        plt.yticks(range(len(content_target_labels)), content_target_labels)
        plt.xlabel("English source content tokens")
        plt.ylabel("German predicted content tokens")
        plt.colorbar(label="Renormalized attention weight")
        plt.tight_layout()
        plt.savefig(content_image_path, dpi=200)
        plt.close()

    print(f"Saved heatmap: {image_path}")
    print(f"Saved weights: {weights_path}")
    if content_attention.size:
        print(f"Saved content-only heatmap: {content_image_path}")
        print(f"Saved content-only weights: {content_weights_path}")
    print(f"  EN:   {sentence}")
    print(f"  PRED: {translation}")
    print(f"  Rows: {', '.join(target_labels)}\n")


def attention_entropy(row):
    row = np.asarray(row)
    row = row / max(row.sum(), 1e-12)
    return float(-(row * np.log(np.maximum(row, 1e-12))).sum())


def content_only_attention(src_tokens, target_labels, attention):
    content_indices = [
        i for i, token in enumerate(src_tokens)
        if token not in ("<SOS>", "<EOS>", "<PAD>")
    ]
    target_indices = [
        i for i, token in enumerate(target_labels)
        if token not in ("<SOS>", "<EOS>", "<PAD>", "<UNK>")
    ]
    if not content_indices or not target_indices:
        return [], [], np.empty((0, 0))

    content_attention = attention[np.ix_(target_indices, content_indices)].copy()
    row_sums = content_attention.sum(axis=1, keepdims=True)
    content_attention = content_attention / np.maximum(row_sums, 1e-12)
    content_tokens = [src_tokens[i] for i in content_indices]
    content_target_labels = [target_labels[i] for i in target_indices]
    return content_tokens, content_target_labels, content_attention


def print_attention_diagnostics(sentence, translation, src_tokens, target_labels, attention):
    eos_idx = src_tokens.index("<EOS>") if "<EOS>" in src_tokens else None
    sos_idx = src_tokens.index("<SOS>") if "<SOS>" in src_tokens else None
    content_indices = [
        i for i, token in enumerate(src_tokens)
        if token not in ("<SOS>", "<EOS>", "<PAD>")
    ]

    print("\nAttention diagnostics")
    print(f"  EN:   {sentence}")
    print(f"  PRED: {translation}")
    print(f"  Source tokens: {' | '.join(src_tokens)}")

    eos_masses = []
    sos_masses = []
    entropies = []
    content_masses = []

    for target, row in zip(target_labels, attention):
        row = row / max(row.sum(), 1e-12)
        top_indices = np.argsort(-row)[:3]
        top_sources = ", ".join(
            f"{src_tokens[i]}={row[i]:.2f}" for i in top_indices
        )

        entropy = attention_entropy(row)
        max_entropy = np.log(len(row)) if len(row) > 1 else 1.0
        normalized_entropy = entropy / max(max_entropy, 1e-12)
        eos_mass = float(row[eos_idx]) if eos_idx is not None else 0.0
        sos_mass = float(row[sos_idx]) if sos_idx is not None else 0.0
        content_mass = float(row[content_indices].sum()) if content_indices else 0.0

        eos_masses.append(eos_mass)
        sos_masses.append(sos_mass)
        entropies.append(normalized_entropy)
        content_masses.append(content_mass)

        print(
            f"  target {target:>12s}: "
            f"top [{top_sources}]  "
            f"EOS={eos_mass:.2f}  content={content_mass:.2f}  "
            f"entropy={normalized_entropy:.2f}"
        )

    avg_eos = float(np.mean(eos_masses)) if eos_masses else 0.0
    avg_sos = float(np.mean(sos_masses)) if sos_masses else 0.0
    avg_content = float(np.mean(content_masses)) if content_masses else 0.0
    avg_entropy = float(np.mean(entropies)) if entropies else 0.0

    print(
        "  Summary: "
        f"avg EOS mass={avg_eos:.2f}, "
        f"avg SOS mass={avg_sos:.2f}, "
        f"avg content mass={avg_content:.2f}, "
        f"avg normalized entropy={avg_entropy:.2f}"
    )
    if avg_eos > 0.45:
        print(
            "  Note: source <EOS> dominates this map. The decoder is likely using the "
            "final encoder state as a sentence summary, so the heatmap may be less "
            "word-aligned than expected."
        )
        print(
            "  Tip: inspect the matching *_content_only.png heatmap to see attention "
            "renormalized over real source and predicted target words only."
        )
    elif avg_entropy > 0.80:
        print(
            "  Note: attention is diffuse. The model is spreading probability across "
            "many source tokens instead of making sharp alignments."
        )
    elif avg_content > 0.60:
        print(
            "  Note: most attention mass is on source content words, so this example "
            "is a better candidate for report-friendly alignment visualization."
        )


def print_fixed_examples(model, examples, eng_vocab, ger_idx2word, max_len, device):
    print("\n-- Manual Demo Examples --")
    print("These are not used for validation metrics.")
    for sentence in examples:
        translation, _, target_labels, _ = translate_with_attention(
            model,
            sentence,
            eng_vocab,
            ger_idx2word,
            max_len,
            device,
        )
        print(f"EN:   {sentence}")
        print(f"PRED: {translation}")
        print(f"RAW:  {' '.join(target_labels)}\n")


def print_random_examples(model, pairs, eng_vocab, ger_idx2word, max_len, device, count, seed):
    rng = random.Random(seed)
    examples = rng.sample(pairs, k=min(count, len(pairs)))

    print("\n-- Random Test Examples --")
    for eng, ger in examples:
        source = " ".join(eng)
        target = " ".join(ger)
        translation, _, target_labels, _ = translate_with_attention(
            model,
            source,
            eng_vocab,
            ger_idx2word,
            max_len,
            device,
        )
        print(f"EN:   {source}")
        print(f"REF:  {target}")
        print(f"PRED: {translation}")
        print(f"RAW:  {' '.join(target_labels)}\n")


def select_test_sentences(pairs, count, seed):
    rng = random.Random(seed)
    examples = rng.sample(pairs, k=min(count, len(pairs)))
    return [" ".join(eng) for eng, ger in examples]


def main():
    parser = argparse.ArgumentParser(description="Validate a trained LSTM seq2seq attention model.")
    parser.add_argument("--data", default="deu.txt")
    parser.add_argument("--model", default="seq2seq_attention.pt")
    parser.add_argument("--english-vocab", default="english_vocab.json")
    parser.add_argument("--german-vocab", default="german_vocab.json")
    parser.add_argument("--english-embeddings", default="english_embeddings.npy")
    parser.add_argument("--german-embeddings", default="german_embeddings.npy")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-examples", type=int, default=10)
    parser.add_argument("--heatmap-dir", default="attention_maps")
    parser.add_argument(
        "--heatmap-count",
        type=int,
        default=5,
        help="Number of held-out test examples to use for heatmaps when --heatmap-sentences is omitted.",
    )
    parser.add_argument(
        "--heatmap-sentences",
        nargs="*",
        default=None,
        help="Optional manual heatmap sentences. If omitted, heatmaps are sampled from the held-out test split.",
    )
    parser.add_argument(
        "--manual-examples",
        nargs="*",
        default=None,
        help="Optional extra sentences to translate. Metrics and random examples still use the held-out test split.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng_vocab = load_vocab(args.english_vocab)
    ger_vocab = load_vocab(args.german_vocab)
    ger_idx2word = {v: k for k, v in ger_vocab.items()}
    eng_embeddings = np.load(args.english_embeddings)
    ger_embeddings = np.load(args.german_embeddings)

    pairs = load_pairs(args.data, args.max_len)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        args.val_split,
        args.test_split,
        args.seed,
        args.pair_limit,
    )

    print(f"Loaded pairs <= {args.max_len} tokens: {len(pairs):,}")
    print(f"Split: train {len(train_pairs):,}, val {len(val_pairs):,}, test {len(test_pairs):,}")
    print(f"Device: {device}")

    test_dataset = TranslationDataset(test_pairs, eng_vocab, ger_vocab, args.max_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args, eng_vocab, ger_vocab, eng_embeddings, ger_embeddings, device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Attention model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    test_loss = evaluate_loss(model, test_loader, criterion, len(ger_vocab), device)
    bleu = evaluate_bleu(model, test_loader, ger_idx2word, args.max_len, device)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f} ({bleu * 100:.2f}%)")

    if args.random_examples > 0:
        print_random_examples(
            model,
            test_pairs,
            eng_vocab,
            ger_idx2word,
            args.max_len,
            device,
            args.random_examples,
            args.seed,
        )

    if args.manual_examples:
        print_fixed_examples(model, args.manual_examples, eng_vocab, ger_idx2word, args.max_len, device)

    print("\n-- Attention Heatmaps --")
    if args.heatmap_sentences:
        heatmap_sentences = args.heatmap_sentences
        print("Using manually provided heatmap sentences. These are not used for validation metrics.")
    else:
        heatmap_sentences = select_test_sentences(test_pairs, args.heatmap_count, args.seed + 1)
        print("Using held-out test split examples for heatmaps.")

    for i, sentence in enumerate(heatmap_sentences, start=1):
        save_attention_heatmap(
            sentence,
            model,
            eng_vocab,
            ger_idx2word,
            args.max_len,
            device,
            args.heatmap_dir,
            f"attention_{i:02d}",
        )


if __name__ == "__main__":
    main()
