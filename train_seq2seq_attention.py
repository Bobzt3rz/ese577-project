import json
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


# Config
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.25
BATCH_SIZE = 256
EPOCHS = 25
LR = 0.001
TEACHER_FORCE_START = 0.75
TEACHER_FORCE_END = 0.25
EARLY_STOP_PATIENCE = 6
MAX_LEN = 15
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
PAIR_LIMIT = None  # set to an int like 50000 for quick experiments
SEED = 42
CHECKPOINT_PATH = "seq2seq_attention.pt"
ATTENTION_HEATMAP_PATH = "attention_heatmap.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_IDS = {PAD, SOS, EOS, UNK}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 1. Load vocab and CBOW embeddings
print("Loading vocab and embeddings...")
with open("english_vocab.json", encoding="utf-8") as f:
    eng_vocab = json.load(f)
with open("german_vocab.json", encoding="utf-8") as f:
    ger_vocab = json.load(f)

eng_embeddings = np.load("english_embeddings.npy")
ger_embeddings = np.load("german_embeddings.npy")
EMBEDDING_DIM = eng_embeddings.shape[1]

if eng_embeddings.shape[1] != ger_embeddings.shape[1]:
    raise ValueError("English and German embeddings must have the same dimension.")

eng_idx2word = {v: k for k, v in eng_vocab.items()}
ger_idx2word = {v: k for k, v in ger_vocab.items()}

print(f"English vocab: {len(eng_vocab):,}  German vocab: {len(ger_vocab):,}")
print(f"Embedding dim: {EMBEDDING_DIM}")


# 2. Data loading and preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def encode(tokens, vocab, max_len=MAX_LEN):
    indices = [vocab.get(token, UNK) for token in tokens[:max_len]]
    indices = [SOS] + indices + [EOS]
    indices += [PAD] * (max_len + 2 - len(indices))
    return indices


def encoded_len(tokens, max_len=MAX_LEN):
    return min(len(tokens), max_len) + 2


print("Loading dataset...")
pairs = []
with open("deu.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            eng = preprocess(parts[0])
            ger = preprocess(parts[1])
            if eng and ger and len(eng) <= MAX_LEN and len(ger) <= MAX_LEN:
                pairs.append((eng, ger))

print(f"Pairs after length filter (max {MAX_LEN}): {len(pairs):,}")
random.shuffle(pairs)

if PAIR_LIMIT is not None:
    pairs = pairs[:PAIR_LIMIT]
    print(f"Using first {len(pairs):,} shuffled pairs for this run")

test_size = int(len(pairs) * TEST_SPLIT)
val_size = int(len(pairs) * VAL_SPLIT)
train_size = len(pairs) - val_size - test_size

train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:train_size + val_size]
test_pairs = pairs[train_size + val_size:]
print(f"Train: {len(train_pairs):,}   Val: {len(val_pairs):,}   Test: {len(test_pairs):,}")


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.src = [torch.tensor(encode(eng, eng_vocab)) for eng, ger in pairs]
        self.tgt = [torch.tensor(encode(ger, ger_vocab)) for eng, ger in pairs]
        self.src_lens = [encoded_len(eng) for eng, ger in pairs]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_lens[idx]


train_dataset = TranslationDataset(train_pairs)
val_dataset = TranslationDataset(val_pairs)
test_dataset = TranslationDataset(test_pairs)
loader_kwargs = {"batch_size": BATCH_SIZE, "pin_memory": torch.cuda.is_available()}
train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, **loader_kwargs)
test_loader = DataLoader(test_dataset, **loader_kwargs)


# 3. LSTM encoder with outputs retained for attention
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.embedding.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
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
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, src_len, hidden]
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.embedding.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.dropout = nn.Dropout(DROPOUT)
        self.attention = DotAttention()
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
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
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, teacher_force_ratio):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab = self.decoder.linear.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab, device=DEVICE)

        encoder_outputs, h, c = self.encoder(src, src_lens)
        src_mask = src != PAD
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, h, c, _ = self.decoder(input_token, h, c, encoder_outputs, src_mask)
            outputs[:, t, :] = pred
            use_teacher = random.random() < teacher_force_ratio
            input_token = tgt[:, t] if use_teacher else pred.argmax(dim=1)

        return outputs


encoder = Encoder(len(eng_vocab), EMBEDDING_DIM, HIDDEN_DIM, eng_embeddings).to(DEVICE)
decoder = AttentionDecoder(len(ger_vocab), EMBEDDING_DIM, HIDDEN_DIM, ger_embeddings).to(DEVICE)
model = Seq2SeqAttention(encoder, decoder).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nAttention model parameters: {total_params:,}")
print(f"Training on: {DEVICE}")


def teacher_force_ratio(epoch):
    if EPOCHS == 1:
        return TEACHER_FORCE_END
    progress = epoch / (EPOCHS - 1)
    return TEACHER_FORCE_START + progress * (TEACHER_FORCE_END - TEACHER_FORCE_START)


def run_epoch(loader, train, teacher_ratio=0.0):
    model.train(train)
    total_loss = 0.0

    for src, tgt, src_lens in loader:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        src_lens = src_lens.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            output = model(src, src_lens, tgt, teacher_ratio)
            output = output[:, 1:, :].reshape(-1, len(ger_vocab))
            gold = tgt[:, 1:].reshape(-1)
            loss = criterion(output, gold)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


print("\nTraining attention model...")
best_val_loss = math.inf
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    ratio = teacher_force_ratio(epoch)
    train_loss = run_epoch(train_loader, train=True, teacher_ratio=ratio)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        saved = "  saved"
    else:
        epochs_without_improvement += 1
        saved = ""

    lr = optimizer.param_groups[0]["lr"]
    print(
        f"  Epoch {epoch + 1}/{EPOCHS}  "
        f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
        f"tf: {ratio:.2f}  lr: {lr:.6f}{saved}"
    )

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"  Early stopping after {EARLY_STOP_PATIENCE} epochs without val improvement")
        break

print(f"\nLoaded best checkpoint from epoch {best_epoch} (val loss {best_val_loss:.4f})")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))


def translate(sentence, max_len=MAX_LEN, return_attention=False):
    model.eval()
    with torch.no_grad():
        tokens = preprocess(sentence)
        indices = encode(tokens, eng_vocab, max_len)
        src = torch.tensor(indices).unsqueeze(0).to(DEVICE)
        src_lens = torch.tensor([encoded_len(tokens, max_len)]).to(DEVICE)
        encoder_outputs, h, c = encoder(src, src_lens)
        src_mask = src != PAD
        input_t = torch.tensor([SOS], device=DEVICE)

        translated = []
        attention_rows = []
        for _ in range(max_len):
            pred, h, c, weights = decoder(input_t, h, c, encoder_outputs, src_mask)
            top_idx = pred.argmax(dim=1)
            word = ger_idx2word.get(top_idx.item(), "<UNK>")
            attention_rows.append(weights.squeeze(0).detach().cpu().numpy())
            if word == "<EOS>":
                break
            if word not in ("<PAD>", "<SOS>", "<UNK>"):
                translated.append(word)
            input_t = top_idx

    if return_attention:
        visible_src = ["<SOS>"] + tokens[:max_len] + ["<EOS>"]
        attention = np.array(attention_rows)[:, :len(visible_src)] if attention_rows else np.empty((0, 0))
        return " ".join(translated), visible_src, translated, attention
    return " ".join(translated)


print("\n-- Example Translations With Attention --")
examples = [
    "I am hungry.",
    "Good morning.",
    "Where is the station?",
    "Tom is a good student.",
    "She loves to read books.",
    "I do not understand.",
]
for sent in examples:
    print(f"  EN: {sent}")
    print(f"  DE: {translate(sent)}\n")


test_loss = run_epoch(test_loader, train=False)
print(f"\nTest loss: {test_loss:.4f}")

print("Computing BLEU score on test set...")
model.eval()
references = []
hypotheses = []

with torch.no_grad():
    for src, tgt, src_lens in test_loader:
        src = src.to(DEVICE)
        src_lens = src_lens.to(DEVICE)
        batch_size = src.shape[0]
        encoder_outputs, h, c = encoder(src, src_lens)
        src_mask = src != PAD
        input_t = torch.full((batch_size,), SOS, dtype=torch.long, device=DEVICE)

        decoded = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        for _ in range(MAX_LEN):
            pred, h, c, _ = decoder(input_t, h, c, encoder_outputs, src_mask)
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

        for i in range(batch_size):
            ref_indices = tgt[i].tolist()
            ref_words = [
                ger_idx2word.get(idx, "<UNK>")
                for idx in ref_indices
                if idx not in SPECIAL_IDS
            ]
            references.append([ref_words])
            hypotheses.append(decoded[i])

smoother = SmoothingFunction().method1
bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)
print(f"\nBLEU Score: {bleu:.4f}  ({bleu * 100:.2f}%)")


def save_attention_heatmap(sentence):
    translation, src_tokens, tgt_tokens, attention = translate(sentence, return_attention=True)
    if attention.size == 0 or not tgt_tokens:
        print("Skipping attention heatmap because no target tokens were produced.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        np.save("attention_weights.npy", attention)
        print("matplotlib is unavailable; saved attention_weights.npy instead.")
        return

    plt.figure(figsize=(max(6, len(src_tokens) * 0.8), max(4, len(tgt_tokens) * 0.45)))
    plt.imshow(attention[:len(tgt_tokens)], aspect="auto", cmap="viridis")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    plt.xlabel("English source tokens")
    plt.ylabel("German output tokens")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(ATTENTION_HEATMAP_PATH, dpi=200)
    plt.close()
    print(f"Saved attention heatmap to {ATTENTION_HEATMAP_PATH}")
    print(f"Heatmap sentence: {sentence}")
    print(f"Heatmap translation: {translation}")


save_attention_heatmap("Where is the station?")
