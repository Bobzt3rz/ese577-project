import re
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 256
NUM_LAYERS     = 2
DROPOUT        = 0.25
BATCH_SIZE     = 256
EPOCHS         = 25
LR             = 0.001
TEACHER_FORCE_START = 0.75
TEACHER_FORCE_END   = 0.25
EARLY_STOP_PATIENCE = 6
MAX_LEN        = 15        # cap sentence length for training
VAL_SPLIT      = 0.10
TEST_SPLIT     = 0.10
PAIR_LIMIT     = None      # set to an int like 50000 for quick experiments
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD, SOS, EOS, UNK = 0, 1, 2, 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── 1. Load Vocab + Embeddings ────────────────────────────────────────────────
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

# ── 2. Load + Preprocess Dataset ──────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def encode(tokens, vocab, max_len=MAX_LEN):
    """Convert tokens to indices, add SOS/EOS, pad to max_len+2."""
    indices = [vocab.get(t, UNK) for t in tokens[:max_len]]
    indices = [SOS] + indices + [EOS]
    # pad
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
            # keep pairs where both sides are within MAX_LEN
            if eng and ger and len(eng) <= MAX_LEN and len(ger) <= MAX_LEN:
                pairs.append((eng, ger))

print(f"Pairs after length filter (max {MAX_LEN}): {len(pairs):,}")

# shuffle and split
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

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.src = [torch.tensor(encode(e, eng_vocab)) for e, g in pairs]
        self.tgt = [torch.tensor(encode(g, ger_vocab)) for e, g in pairs]
        self.src_lens = [encoded_len(e) for e, g in pairs]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_lens[idx]

train_dataset = TranslationDataset(train_pairs)
val_dataset = TranslationDataset(val_pairs)
test_dataset  = TranslationDataset(test_pairs)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available())
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available())

# ── 4. Encoder ────────────────────────────────────────────────────────────────
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

    def forward(self, x, lengths):
        # x: [batch, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch, seq_len, emb_dim]
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h, c) = self.lstm(packed)           # h,c: [layers, batch, hidden_dim]
        return h, c

# ── 5. Decoder ────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
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
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c):
        # x: [batch] — one token at a time
        embedded     = self.dropout(self.embedding(x.unsqueeze(1)))  # [batch, 1, emb_dim]
        out, (h, c)  = self.lstm(embedded, (h, c))         # [batch, 1, hidden]
        pred         = self.linear(out.squeeze(1))         # [batch, vocab_size]
        return pred, h, c

# ── 6. Seq2Seq Wrapper ────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, teacher_force_ratio):
        batch_size = src.shape[0]
        tgt_len    = tgt.shape[1]
        tgt_vocab  = self.decoder.linear.out_features

        # store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab).to(DEVICE)

        # encode source sentence → context vector
        h, c = self.encoder(src, src_lens)

        # first decoder input is <SOS>
        input_token = tgt[:, 0]   # [batch]

        for t in range(1, tgt_len):
            pred, h, c = self.decoder(input_token, h, c)
            outputs[:, t, :] = pred

            # teacher forcing: use real target or model's prediction
            use_teacher = random.random() < teacher_force_ratio
            input_token = tgt[:, t] if use_teacher else pred.argmax(dim=1)

        return outputs

# ── 7. Initialize Models ──────────────────────────────────────────────────────
encoder = Encoder(len(eng_vocab), EMBEDDING_DIM, HIDDEN_DIM, eng_embeddings).to(DEVICE)
decoder = Decoder(len(ger_vocab), EMBEDDING_DIM, HIDDEN_DIM, ger_embeddings).to(DEVICE)
model   = Seq2Seq(encoder, decoder).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")
print(f"Training on: {DEVICE}")

# ── 8. Training Loop ──────────────────────────────────────────────────────────
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

print("\nTraining...")
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
        torch.save(model.state_dict(), "seq2seq.pt")
        saved = "  saved"
    else:
        epochs_without_improvement += 1
        saved = ""

    lr = optimizer.param_groups[0]["lr"]
    print(
        f"  Epoch {epoch+1}/{EPOCHS}  "
        f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
        f"tf: {ratio:.2f}  lr: {lr:.6f}{saved}"
    )

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"  Early stopping after {EARLY_STOP_PATIENCE} epochs without val improvement")
        break

print(f"\nLoaded best checkpoint from epoch {best_epoch} (val loss {best_val_loss:.4f})")
model.load_state_dict(torch.load("seq2seq.pt", map_location=DEVICE))

# ── 9. Translation Function ───────────────────────────────────────────────────
def translate(sentence, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        tokens  = preprocess(sentence)
        indices = encode(tokens, eng_vocab, max_len)
        src     = torch.tensor(indices).unsqueeze(0).to(DEVICE)  # [1, seq_len]
        src_lens = torch.tensor([encoded_len(tokens, max_len)]).to(DEVICE)

        h, c    = encoder(src, src_lens)
        input_t = torch.tensor([SOS]).to(DEVICE)

        translated = []
        for _ in range(max_len):
            pred, h, c = decoder(input_t, h, c)
            top_idx    = pred.argmax(dim=1)
            word       = ger_idx2word.get(top_idx.item(), "<UNK>")
            if word == "<EOS>":
                break
            if word not in ("<PAD>", "<SOS>", "<UNK>"):
                translated.append(word)
            input_t = top_idx

    return " ".join(translated)

# ── 10. Example Translations ──────────────────────────────────────────────────
print("\n-- Example Translations --")
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

# ── 11. BLEU Score ────────────────────────────────────────────────────────────
test_loss = run_epoch(test_loader, train=False)
print(f"\nTest loss: {test_loss:.4f}")

print("Computing BLEU score on test set...")
model.eval()
references  = []
hypotheses  = []

with torch.no_grad():
    for src, tgt, src_lens in test_loader:
        src = src.to(DEVICE)
        src_lens = src_lens.to(DEVICE)
        batch_size = src.shape[0]
        h, c = encoder(src, src_lens)
        input_t = torch.full((batch_size,), SOS, dtype=torch.long).to(DEVICE)

        decoded = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        for _ in range(MAX_LEN):
            pred, h, c = decoder(input_t, h, c)
            top_idx    = pred.argmax(dim=1)
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
            ref_words   = [ger_idx2word.get(idx, "<UNK>")
                           for idx in ref_indices
                           if idx not in (PAD, SOS, EOS, UNK)]
            references.append([ref_words])
            hypotheses.append(decoded[i])

smoother = SmoothingFunction().method1
bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)
print(f"\nBLEU Score: {bleu:.4f}  ({bleu*100:.2f}%)")
