import re
import json
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
WINDOW_SIZE = 2
EPOCHS = 15
BATCH_SIZE = 2048
MIN_FREQ = 2
LR = 0.001

# ── 1. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

print("Loading dataset...")
english_sentences = []
german_sentences  = []

with open("deu.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            eng = preprocess(parts[0])
            ger = preprocess(parts[1])
            if eng and ger:
                english_sentences.append(eng)
                german_sentences.append(ger)

print(f"Loaded {len(english_sentences):,} sentence pairs")

# ── 2. Build Vocabularies ─────────────────────────────────────────────────────
def build_vocab(sentences, min_freq=1):
    counter = Counter(w for s in sentences for w in s)
    # special tokens first
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

print("\nBuilding vocabularies...")
english_vocab = build_vocab(english_sentences, MIN_FREQ)
german_vocab  = build_vocab(german_sentences,  MIN_FREQ)

print(f"English vocab size: {len(english_vocab):,}")
print(f"German vocab size:  {len(german_vocab):,}")

# save vocabularies - needed later for seq2seq
with open("english_vocab.json", "w", encoding="utf-8") as f:
    json.dump(english_vocab, f, ensure_ascii=False)
with open("german_vocab.json", "w", encoding="utf-8") as f:
    json.dump(german_vocab, f, ensure_ascii=False)
print("Saved english_vocab.json and german_vocab.json")

# ── 3. CBOW Dataset ───────────────────────────────────────────────────────────
def generate_cbow_pairs(sentences, vocab, window_size=2):
    """For each center word, collect its context words as input."""
    contexts = []
    centers  = []
    unk_idx  = vocab["<UNK>"]

    for sentence in sentences:
        indices = [vocab.get(w, unk_idx) for w in sentence]
        for i in range(len(indices)):
            left  = indices[max(0, i - window_size):i]
            right = indices[i + 1: i + 1 + window_size]
            context = left + right
            if not context:
                continue
            # pad context to fixed size (2 * window_size) so we can batch
            while len(context) < 2 * window_size:
                context.append(vocab["<PAD>"])
            contexts.append(context)
            centers.append(indices[i])

    return torch.tensor(contexts, dtype=torch.long), \
           torch.tensor(centers,  dtype=torch.long)

class CBOWDataset(Dataset):
    def __init__(self, contexts, centers):
        self.contexts = contexts
        self.centers  = centers
    def __len__(self):
        return len(self.centers)
    def __getitem__(self, idx):
        return self.contexts[idx], self.centers[idx]

# ── 4. CBOW Model ─────────────────────────────────────────────────────────────
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear     = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context: [batch, 2*window_size]
        embeds = self.embeddings(context)   # [batch, context_len, emb_dim]
        avg    = embeds.mean(dim=1)         # [batch, emb_dim]
        return self.linear(avg)             # [batch, vocab_size]

# ── 5. Training Function ──────────────────────────────────────────────────────
def train_cbow(sentences, vocab, lang_name, embedding_dim=EMBEDDING_DIM,
               window_size=WINDOW_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{lang_name}] Generating CBOW pairs (this may take a minute)...")

    contexts, centers = generate_cbow_pairs(sentences, vocab, window_size)
    print(f"[{lang_name}] Total training pairs: {len(centers):,}")

    dataset    = CBOWDataset(contexts, centers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = CBOW(len(vocab), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"[{lang_name}] Training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for ctx_batch, ctr_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            ctr_batch = ctr_batch.to(device)

            output = model(ctx_batch)
            loss   = criterion(output, ctr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

    # extract and return the embedding matrix as numpy
    embeddings = model.embeddings.weight.data.cpu().numpy()
    return embeddings

# ── 6. Train Both ─────────────────────────────────────────────────────────────
english_embeddings = train_cbow(english_sentences, english_vocab, "English")
german_embeddings  = train_cbow(german_sentences,  german_vocab,  "German")

# ── 7. Save Embeddings ────────────────────────────────────────────────────────
np.save("english_embeddings.npy", english_embeddings)
np.save("german_embeddings.npy",  german_embeddings)

print("\nSaved:")
print(f"  english_embeddings.npy  shape: {english_embeddings.shape}")
print(f"  german_embeddings.npy   shape: {german_embeddings.shape}")

# ── 8. Quick Sanity Check ─────────────────────────────────────────────────────
def nearest_words(word, vocab, embeddings, top_n=5):
    idx_to_word = {v: k for k, v in vocab.items()}
    unk_idx     = vocab["<UNK>"]
    idx         = vocab.get(word, unk_idx)
    vec         = torch.tensor(embeddings[idx]).unsqueeze(0)
    all_vecs    = torch.tensor(embeddings)
    sims        = torch.cosine_similarity(vec, all_vecs)
    top_indices = sims.topk(top_n + 1).indices[1:]  # skip word itself
    return [idx_to_word[i.item()] for i in top_indices]

print("\n-- Nearest words sanity check --")
for word in ["good", "house", "go"]:
    print(f"  English '{word}': {nearest_words(word, english_vocab, english_embeddings)}")

for word in ["gut", "haus", "gehen"]:
    print(f"  German  '{word}': {nearest_words(word, german_vocab, german_embeddings)}")