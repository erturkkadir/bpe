# -*- coding: utf-8 -*-
"""
TOKEN NEDİR? —
=================================================
Yunus Emre dörtlüğü üzerinde SIFIRDAN bir BPE tokenizer eğitir:
  1) metni harflere böl
  2) en sık geçen ikiliyi yapıştır
  3) doyana kadar tekrarla
  4) hiç görülmemiş kelimeyi tokenla

Çalıştır:  python3 bpe_demo.py            (her bölüm Enter ile ilerler)
           python3 bpe_demo.py --hizli    (duraklamadan akar — prova için)

kütüphane YOK — sadece Python. Çekirdek algoritma ~15 satır (aşağıda
"ÇEKİRDEK" bloğu); gerçek tokenizer'lar (GPT'ninki dahil) aynı fikrin
internet ölçeğinde koşulmuş hâlidir.
"""
import sys
from collections import Counter

# ---- Veri: Yunus Emre dörtlüğü -----------------------------------------
DORTLUK = [
    "gelin tanış olalım",
    "işin kolayın tutalım",
    "sevelim sevilelim",
    "dünya kimseye kalmaz",
]
MERGE_SAYISI = 12          # kaç birleştirme adımı (sözlük ne kadar büyüsün)
TEST_KELIMELER = ["sevgilim", "sevmelim", "tutalım"]   # eğitimde YOK (tutalım hariç)

# ---- Terminal süsleri ---------------------------------------------------
RENKLER = ["\033[44m", "\033[45m", "\033[46m", "\033[42m", "\033[43m", "\033[41m"]
SIFIRLA, KALIN, SOLUK = "\033[0m", "\033[1m", "\033[2m"
HIZLI = "--hizli" in sys.argv

def bekle(mesaj="devam için Enter"):
    if not HIZLI:
        input(f"{SOLUK}   [{mesaj}]{SIFIRLA}")
    print()

def boyali(parcalar):
    """Token listesini bitişik renkli bloklar hâlinde tek satır yazar."""
    return "".join(f"{RENKLER[i % len(RENKLER)]} {p} {SIFIRLA}"
                   for i, p in enumerate(parcalar))

def baslik(no, metin):
    print(f"\n{KALIN}━━ {no}. {metin} {'━' * max(4, 46 - len(metin))}{SIFIRLA}\n")


# ════════════════════════ ÇEKİRDEK: BPE ═════════════════════════════════
def ikilileri_say(kelimeler):
    """Tüm kelimelerdeki yan yana parça ikililerini sayar."""
    sayac = Counter()
    for parcalar, adet in kelimeler.items():
        for ikili in zip(parcalar, parcalar[1:]):
            sayac[ikili] += adet
    return sayac

def yapistir(kelimeler, a, b):
    """Her kelimede yan yana gelen (a,b) ikilisini tek parça 'ab' yapar."""
    yeni = {}
    for parcalar, adet in kelimeler.items():
        p, i = [], 0
        while i < len(parcalar):
            if i + 1 < len(parcalar) and parcalar[i] == a and parcalar[i + 1] == b:
                p.append(a + b); i += 2
            else:
                p.append(parcalar[i]); i += 1
        yeni[tuple(p)] = adet
    return yeni

def tokenla(kelime, birlesmeler):
    """Yeni bir kelimeyi, öğrenilen birleştirmeleri SIRAYLA uygulayarak böler."""
    parcalar = list(kelime)
    for a, b in birlesmeler:
        parcalar = list(yapistir({tuple(parcalar): 1}, a, b))[0]
    return list(parcalar)
# ═════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    metin = " ".join(DORTLUK)

    # ---- 1) Başlangıç: metin = harfler -----------------------------------
    baslik(1, "Bilgisayarın gördüğü: sadece karakterler")
    for dize in DORTLUK:
        print("   " + dize)
    harfler = sorted(set(metin.replace(" ", "")))
    print(f"\n   odak dize, harf harf:  {boyali(list(DORTLUK[2].replace(' ', '·')))}")
    print(f"\n   başlangıç sözlüğü = {len(harfler)} harf: {' '.join(harfler)}")
    bekle()

    # kelime -> frekans; her kelime harf demeti olarak başlar
    sayilar = Counter(metin.split())
    kelimeler = {tuple(k): n for k, n in sayilar.items()}

    # ---- 2) Eğitim: en sık ikiliyi yapıştır, tekrarla ---------------------
    baslik(2, "BPE eğitimi: en sık ikiliyi yapıştır")
    birlesmeler, sozluk = [], list(harfler)
    for adim in range(1, MERGE_SAYISI + 1):
        sayac = ikilileri_say(kelimeler)
        (a, b), kac = sayac.most_common(1)[0]
        if kac < 2:
            print("   (tekrar eden ikili kalmadı — eğitim doydu)")
            break
        kelimeler = yapistir(kelimeler, a, b)
        birlesmeler.append((a, b))
        sozluk.append(a + b)
        print(f"   adım {adim:2d}:  '{a}' + '{b}'  →  "
              f"{KALIN}'{a+b}'{SIFIRLA}   ({kac} kez geçiyor)")
    print(f"\n   sözlük artık {len(sozluk)} parça "
          f"({len(harfler)} harf + {len(birlesmeler)} öğrenilen)")
    bekle()

    # ---- 3) Sonuç: dörtlük token'larıyla ----------------------------------
    baslik(3, "Dörtlük, öğrenilen token'larla")
    for dize in DORTLUK:
        parcalar = []
        for kelime in dize.split():
            parcalar += tokenla(kelime, birlesmeler)
        print("   " + boyali(parcalar))
    print(f"\n   {SOLUK}'sev' kökünü kimse söylemedi — sıklık kendiliğinden buldu.{SIFIRLA}")
    bekle()

    # ---- 4) Asıl test: hiç görülmemiş kelime ------------------------------
    baslik(4, "Genelleme: eğitimde OLMAYAN kelimeler")
    for kelime in TEST_KELIMELER:
        parcalar = tokenla(kelime, birlesmeler)
        print(f"   {kelime:12s} →  {boyali(parcalar)}")
    print(f"\n   kelime-sözlüğü olsaydı: 'sevgilim' → [BİLİNMEYEN]  (okunamazdı)")
    print("   parça-sözlüğüyle: yeni kelime bile bilinen parçalardan kurulur.")
    bekle()

    # ---- 5) Karşılaştırma --------------------------------------------------
    baslik(5, "Üç strateji, tek tablo (bu dörtlük için)")
    n_harf = len(metin.replace(" ", ""))
    n_bpe = sum(len(tokenla(k, birlesmeler)) for k in metin.split())
    n_kelime = len(metin.split())
    print(f"   {'strateji':12s} {'dizi uzunluğu':>14s} {'sözlük boyu':>12s}")
    print(f"   {'-'*12} {'-'*14:>14s} {'-'*12:>12s}")
    print(f"   {'karakter':12s} {n_harf:>14d} {len(harfler):>12d}   dizi upuzun")
    print(f"   {'BPE':12s} {n_bpe:>14d} {len(sozluk):>12d}   orta yol ✓")
    print(f"   {'kelime':12s} {n_kelime:>14d} {'∞?':>12s}   sözlük patlar")
    print(f"\n{KALIN}   Model kelime görmez — token görür.{SIFIRLA}\n")
