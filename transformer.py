# ============================================================
#  TRANSFORMER — SIFIRDAN, SATIR SATIR, TEK GPU
#  Trans0 | Decoder-only (GPT tarzı) karakter seviyesi dil modeli
#
#  Tek bağımlılık: PyTorch. nn.Transformer YOK — her şey elle.
#  Tensör şekilleri: B = batch, T = zaman (bağlam), C = kanal (embedding)
# ============================================================

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Hiperparametreler ----------
batch_size = 64      # aynı anda kaç bağımsız dizi işlenecek (B)
block_size = 128     # bağlam uzunluğu: model en fazla kaç karakter geriye bakar (T)
n_embd     = 128     # embedding boyutu: her karakteri temsil eden vektörün uzunluğu (C)
n_head     = 4       # attention head sayısı (her head C/n_head = 32 boyutlu)
n_layer    = 4       # üst üste kaç transformer bloğu
dropout    = 0.1     # aşırı öğrenmeye karşı rastgele nöron kapatma oranı
max_iters  = 30000   # eğitim adımı sayısı
lr         = 3e-4    # öğrenme hızı (AdamW için klasik değer)
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


# ============================================================
#  BÖLÜM 1 — VERİ & TOKENIZER
#  Fikir: dili sayıya çevirmeden model hiçbir şey yapamaz.
#  İKİ yol var (detay ve kıyas: tokenizer.py — önce onu çalıştır!):
#    'karakter' : küçük sözlük (~88), her kelimeyi harflerden kurar
#    'kelime'   : kısa diziler ama 5000 kelimelik sözlük + <unk> kaybı
# ============================================================

from tokenizer import KarakterTokenizer, KelimeTokenizer

TOKEN_TIP = 'karakter'      # 'karakter' veya 'kelime' — değiştir, farkı gör!

with open('veri.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if TOKEN_TIP == 'karakter':
    tok = KarakterTokenizer(text)
else:
    tok = KelimeTokenizer(text, max_vocab=5000)

vocab_size = tok.vocab_size
encode, decode = tok.encode, tok.decode

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]              # %90 eğitim
val_data   = data[n:]              # %10 doğrulama (modelin ezber yapıp yapmadığını ölçer)


# ============================================================
#  BÖLÜM 2 — BATCH HAZIRLAMA
#  Girdi x: block_size uzunluğunda bir parça.
#  Hedef y: aynı parçanın BİR SAĞA kaymış hali.
#  Yani her konumda görev: "bir sonraki karakteri tahmin et".
# ============================================================

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))     # rastgele başlangıçlar
    x = torch.stack([d[i     : i+block_size    ] for i in ix])  # (B, T)
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])  # (B, T) — 1 kaymış
    return x.to(device), y.to(device)


# ============================================================
#  BÖLÜM 3 — TEK ATTENTION HEAD  ⭐ videonun kalbi
#  Her karakter 3 vektör üretir:
#    Q (query)  : "ben ne arıyorum?"
#    K (key)    : "bende ne var?"
#    V (value)  : "eşleşirsek sana vereceğim bilgi"
#  Q·K benzerliği yüksekse, o karakterin V'sinden çok alınır.
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Nedensel maske: alt üçgen matris. Gelecek görünmez!
        # tril[i][j] = 1 ise i. karakter j. karaktere bakabilir (j <= i)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)                        # (B, T, head_size)
        k = self.key(x)                          # (B, T, head_size)

        # Benzerlik puanları: her karakter, her karakterle karşılaştırılır
        wei = q @ k.transpose(-2, -1)            # (B, T, T)
        wei = wei * k.shape[-1] ** -0.5          # √d ile ölçekle: puanlar çok
                                                 # büyürse softmax tek noktaya çöker

        # Geleceği kapat: j > i olan hücrelere -sonsuz yaz
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)             # (B, T, T) — her satır toplamı 1
        wei = self.dropout(wei)

        v = self.value(x)                        # (B, T, head_size)
        out = wei @ v                            # (B, T, head_size)
        return out                               # = geçmişin ağırlıklı ortalaması


# ============================================================
#  BÖLÜM 4 — MULTI-HEAD ATTENTION
#  Tek head tek tür ilişki öğrenir. n_head paralel head,
#  farklı ilişkiler öğrenir (ünlü uyumu, kelime sınırı, ...).
#  Çıktılar yan yana eklenir ve bir Linear ile karıştırılır.
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)    # head'lerin bilgisini harmanla
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, C)
        return self.dropout(self.proj(out))


# ============================================================
#  BÖLÜM 5 — FEEDFORWARD
#  Attention bilgiyi TOPLAR, FFN o bilgiyi İŞLER.
#  Klasik tarif: 4 kat genişlet -> ReLU -> geri daralt.
#  Her karakter pozisyonu bağımsız işlenir (konuşma bitti, düşünme zamanı).
# ============================================================

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#  BÖLÜM 6 — TRANSFORMER BLOĞU
#  Blok = iletişim (attention) + hesaplama (FFN)
#  İki püf nokta:
#    1) Residual: x + f(x) — gradyan otoyolu, derin ağ eğitilebilir kalır
#    2) LayerNorm önce uygulanır (pre-norm, modern standart)
# ============================================================

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa   = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # karakterler birbiriyle konuşur
        x = x + self.ffwd(self.ln2(x))   # her karakter kendi başına düşünür
        return x


# ============================================================
#  BÖLÜM 7 — GPT MODELİ (her şeyi birleştir)
#  token embedding : "bu karakter NE?"
#  pozisyon embed. : "bu karakter NEREDE?"
#  Attention sırayı bilmez — pozisyonu biz eklemek zorundayız!
# ============================================================

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)   # embedding -> harf puanları

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                            # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb                                          # (B, T, C)
        x = self.blocks(x)                                             # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                       # (B, T, vocab)

        if targets is None:
            return logits, None
        # cross_entropy 2D ister: (B*T, vocab) puanlar, (B*T,) doğru cevaplar
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) — mevcut bağlam. Her adımda 1 karakter üret, sona ekle.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # bağlamı pencereye sığdır
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # sadece SON konumun tahmini
            probs = F.softmax(logits, dim=-1)        # puanlar -> olasılıklar
            idx_next = torch.multinomial(probs, 1)   # olasılığa göre zar at
            idx = torch.cat([idx, idx_next], dim=1)  # üretileni bağlama ekle
        return idx


# ============================================================
#  BÖLÜM 8 — EĞİTİM DÖNGÜSÜ
#  Tarif hep aynı: tahmin et -> hatayı ölç -> gradyan -> güncelle
# ============================================================

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(50)
        for k in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

if __name__ == '__main__':
    model = GPT().to(device)
    print(f"Cihaz: {device} | Tokenizer: {TOKEN_TIP} | Sözlük: {vocab_size} | "
          f"Parametre: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    eval_interval = max_iters // 10        # 10 ara rapor: loss'un seyri görünsün
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"adım {it:4d} | train loss {losses['train']:.3f} | "
                  f"val loss {losses['val']:.3f}")

        x, y = get_batch('train')
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()          # backpropagation — detayı trans2'de!
        optimizer.step()

    # ============================================================
    #  BÖLÜM 9 — ÜRETİM: model artık "Türkçemsi" yazabilir
    # ============================================================
    print("\n--- Modelin ürettiği metin ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
