import argparse
import json
import random
import re

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
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
        _, (h, c) = self.lstm(packed)
        return h, c


class Decoder(nn.Module):
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
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, h, c):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        out, (h, c) = self.lstm(embedded, (h, c))
        pred = self.linear(out.squeeze(1))
        return pred, h, c


class Seq2Seq(nn.Module):
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

        h, c = self.encoder(src, src_lens)
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, h, c = self.decoder(input_token, h, c)
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
    decoder = Decoder(
        len(ger_vocab),
        embedding_dim,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        ger_embeddings,
    )
    return Seq2Seq(encoder, decoder, device).to(device)


def decode_batch(model, src, src_lens, ger_idx2word, max_len, device):
    model.eval()
    batch_size = src.shape[0]
    h, c = model.encoder(src, src_lens)
    input_t = torch.full((batch_size,), SOS, dtype=torch.long, device=device)
    decoded = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        pred, h, c = model.decoder(input_t, h, c)
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


def translate(model, sentence, eng_vocab, ger_idx2word, max_len, device):
    tokens = preprocess(sentence)
    src = torch.tensor(encode(tokens, eng_vocab, max_len)).unsqueeze(0).to(device)
    src_lens = torch.tensor([encoded_len(tokens, max_len)]).to(device)
    decoded = decode_batch(model, src, src_lens, ger_idx2word, max_len, device)
    return " ".join(decoded[0])


def print_random_examples(model, pairs, eng_vocab, ger_idx2word, max_len, device, count, seed):
    rng = random.Random(seed)
    examples = rng.sample(pairs, k=min(count, len(pairs)))

    print("\n-- Random Test Examples --")
    for eng, ger in examples:
        source = " ".join(eng)
        target = " ".join(ger)
        pred = translate(model, source, eng_vocab, ger_idx2word, max_len, device)
        print(f"EN:   {source}")
        print(f"REF:  {target}")
        print(f"PRED: {pred}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate a trained no-attention LSTM seq2seq model.")
    parser.add_argument("--data", default="deu.txt")
    parser.add_argument("--model", default="seq2seq.pt")
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
    parser.add_argument("--random-examples", type=int, default=5)
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
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    test_loss = evaluate_loss(model, test_loader, criterion, len(ger_vocab), device)
    bleu = evaluate_bleu(model, test_loader, ger_idx2word, args.max_len, device)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f} ({bleu * 100:.2f}%)")

    fixed_examples = [
        "I am hungry.",
        "Good morning.",
        "Where is the station?",
        "Tom is a good student.",
        "She loves to read books.",
        "I do not understand.",
    ]
    print("\n-- Fixed Examples --")
    for sentence in fixed_examples:
        print(f"EN:   {sentence}")
        print(f"PRED: {translate(model, sentence, eng_vocab, ger_idx2word, args.max_len, device)}\n")

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


if __name__ == "__main__":
    main()
