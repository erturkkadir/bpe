# ============================================================
#  TOKENIZER — DİLİ SAYIYA ÇEVİRME SANATI
#  Trans0 | İki yaklaşım: KARAKTER ve KELİME seviyesi
#
#  Model harf bilmez, kelime bilmez — sadece sayı bilir.
#  "Metni hangi parçalara bölüp numaralandıracağız?" sorusunun
#  cevabı, modelin kaderini belirler. İşte iki uç:
#
#    KARAKTER : küçük sözlük (~88), uzun diziler, kelime bilgisi yok
#    KELİME   : dev sözlük (on binlerce), kısa diziler,
#               ama görmediği kelimede çaresiz (<unk> sorunu!)
#
#  Gerçek LLM'ler ikisinin ORTASINI kullanır: alt-kelime (BPE).
#  "olalım" -> "ol" + "alım" gibi. O ayrı bir bölümün konusu.
# ============================================================

import re


# ------------------------------------------------------------
#  1) KARAKTER TOKENIZER
#  Alfabeyi metnin kendisinden çıkarır. 5 satırda biter,
#  HİÇBİR kelimeye yabancı kalmaz — her kelime harflerden kurulur.
# ------------------------------------------------------------

class KarakterTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))                       # metindeki tüm farklı karakterler
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # karakter -> sayı
        self.itos = {i: ch for i, ch in enumerate(chars)}   # sayı -> karakter
        self.vocab_size = len(chars)

    def encode(self, s):                # "olalım" -> [66, 61, 50, 61, 85, 62]
        return [self.stoi[c] for c in s]

    def decode(self, ids):              # sayı listesi -> metin
        return ''.join(self.itos[i] for i in ids)


# ------------------------------------------------------------
#  2) KELİME TOKENIZER
#  Her kelimeye bir numara. Kulağa doğal geliyor, ama iki bedeli var:
#    a) Sözlük patlar: 652 KB metinde bile ~50 bin farklı kelime.
#       Türkçe eklerle çoğalır: "ev, evim, evimde, evimdekiler..."
#       hepsi AYRI kelime sayılır!
#    b) <unk> sorunu: sözlükte olmayan kelime "bilinmeyen"e düşer.
#       Model o kelime hakkında HİÇBİR şey bilemez.
#  Çare: sadece en sık geçen max_vocab kelimeyi tut, gerisi <unk>.
# ------------------------------------------------------------

class KelimeTokenizer:
    UNK = '<unk>'                        # sözlük dışı her şeyin ortak adı

    def __init__(self, text, max_vocab=5000):
        kelimeler = self._parcala(text)
        # frekans say: en sık geçenler sözlüğe girer
        freq = {}
        for k in kelimeler:
            freq[k] = freq.get(k, 0) + 1
        siralik = sorted(freq, key=freq.get, reverse=True)[:max_vocab - 1]
        vocab = [self.UNK] + siralik     # id 0 = <unk>
        self.stoi = {k: i for i, k in enumerate(vocab)}
        self.itos = {i: k for i, k in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.toplam_kelime = len(kelimeler)
        # kapsama: metindeki kelimelerin yüzde kaçı sözlükte?
        self.kapsama = sum(freq[k] for k in siralik) / len(kelimeler)

    def _parcala(self, s):
        # \w+ = harf/rakam dizileri (kelimeler), [^\w\s] = noktalama işaretleri
        # Noktalama da ayrı token olur: "olalım." -> ["olalım", "."]
        return re.findall(r"\w+|[^\w\s]", s)

    def encode(self, s):                 # "gelin tanış olalım" -> [1750, 0, 2371]
        unk = self.stoi[self.UNK]
        return [self.stoi.get(k, unk) for k in self._parcala(s)]

    def decode(self, ids):               # kelimeleri boşlukla birleştir
        out = ' '.join(self.itos[i] for i in ids)
        return re.sub(r" ([.,;:!?)])", r"\1", out)   # noktalamayı kelimeye yapıştır


# ------------------------------------------------------------
#  DEMO — iki tokenizer'ı aynı metin üzerinde kıyasla
#  Çalıştır: python3 tokenizer.py
# ------------------------------------------------------------

if __name__ == '__main__':
    with open('veri.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    ornek = "Gelin tanış olalım."

    print("=" * 60)
    print("METİN:", f"{len(text):,} karakter  |  ÖRNEK:", repr(ornek))
    print("=" * 60)

    # --- karakter seviyesi ---
    kt = KarakterTokenizer(text)
    ids = kt.encode(ornek)
    print(f"\n[KARAKTER]  sözlük: {kt.vocab_size} parça")
    print(f"  encode -> {ids}")
    print(f"  {len(ids)} token  |  decode -> {kt.decode(ids)!r}")

    # --- kelime seviyesi ---
    wt = KelimeTokenizer(text, max_vocab=5000)
    ids = wt.encode(ornek)
    print(f"\n[KELİME]    sözlük: {wt.vocab_size:,} parça "
          f"(metindeki tüm farklı kelimeler çok daha fazla!)")
    print(f"  metnin %{wt.kapsama * 100:.1f}'i bu 5000 kelimeyle kapsanıyor")
    print(f"  encode -> {ids}")
    print(f"  {len(ids)} token  |  decode -> {wt.decode(ids)!r}")

    # --- <unk> sorununu canlı göster ---
    nadir = "Kapadokya'daki peribacaları büyüleyiciydi."
    ids = wt.encode(nadir)
    print(f"\n[<unk> SORUNU]  {nadir!r}")
    print(f"  kelime decode -> {wt.decode(ids)!r}")
    print(f"  karakter decode -> {kt.decode(kt.encode(nadir))!r}   (kayıpsız!)")

    print("\nSONUÇ: karakter = küçük sözlük ama uzun dizi,")
    print("       kelime   = kısa dizi ama dev sözlük + <unk> kaybı.")
    print("       Gerçek LLM'ler ortayı seçer: alt-kelime (BPE) — ileriki bölüm!")
