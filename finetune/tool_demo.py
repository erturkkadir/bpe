
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

from istemci import MODEL_ADI, SUNUCU, Model

KUCUK_MODEL = "smollm2:135m"       # S6'daki kötü örnek: araç çağrısını beceremez

HIZLI = "--hizli" in sys.argv
OLLAMA_TURU = "--ollama" in sys.argv

# ---- terminal süsü ------------------------------------------------------
R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
MAVI = "\033[38;5;75m"; YESIL = "\033[38;5;79m"; KIRMIZI = "\033[38;5;203m"
AMBER = "\033[38;5;215m"; MOR = "\033[38;5;141m"


def baslik(n, metin, renk=MAVI):
    print(f"\n{renk}{B}{'━' * 72}{R}")
    print(f"{renk}{B}  {n}. {metin}{R}")
    print(f"{renk}{B}{'━' * 72}{R}\n")


def duraklat(mesaj="devam için Enter"):
    if HIZLI:
        return
    try:
        input(f"\n{DIM}   [{mesaj}]{R}")
    except EOFError:
        pass


# ---- ARAÇLAR: gerçekten çalışan fonksiyonlar ----------------------------
def kur_getir(para: str) -> str:
    """Bir para biriminin güncel TL karşılığını döndürür (gerçek servis)."""
    url = f"https://open.er-api.com/v6/latest/{para.upper()}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            veri = json.loads(r.read())
        try_kur = veri["rates"]["TRY"]
        tarih = veri.get("time_last_update_utc", "")[:16]
        return f"1 {para.upper()} = {try_kur:.2f} TL  (kaynak: open.er-api.com, {tarih})"
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
        # Dürüstlük: servise ulaşamadıysak bunu SAKLAMIYORUZ, ekranda söylüyoruz.
        print(f"{KIRMIZI}   [!] servise ulaşılamadı ({e.__class__.__name__}) — "
              f"yedek sabit değer kullanılıyor{R}")
        return f"1 {para.upper()} = 41.83 TL  (YEDEK SABİT DEĞER — servis kapalı)"


def saat() -> str:
    """Şu anki tarih ve saat."""
    return datetime.now().strftime("%d.%m.%Y %H:%M")


ARACLAR = {"kur_getir": kur_getir, "saat": saat}

# ---- Modele verilen TARİF (kod değil, sadece tarif) ---------------------
TARIFLER = [
    {
        "ad": "kur_getir",
        "ne_yapar": "Bir para biriminin guncel Turk Lirasi karsiligini verir. "
                    "Doviz kuru sorulan her durumda bu araci cagir; kendi "
                    "bilginden tahmin yurutme.",
        "parametreler": {
            "para": {"tip": "string",
                     "aciklama": "Para birimi kodu, ornegin USD, EUR, GBP"},
        },
        "zorunlu": ["para"],
    },
    {
        "ad": "saat",
        "ne_yapar": "Su anki tarih ve saati verir.",
        "parametreler": {},
    },
]

REPO = "github.com/erturkkadir/bpe/tree/main/finetune"   # theme.REPO + REPO_DIR

SORU = "Bugün 1 dolar kaç TL?"
SORU2 = "Dolar ve euro şu an kaç TL? Bir de saati söyle."


def blok_yaz(bloklar):
    """Modelin ürettiği içerik bloklarını okunur biçimde bas."""
    for b in bloklar:
        if b["tip"] == "metin":
            print(f"{MAVI}   model:{R} {b['metin']}")
        elif b["tip"] == "cagri":
            cagri = json.dumps({"tool": b["ad"], "args": b["girdi"]},
                               ensure_ascii=False)
            print(f"{AMBER}   model, ARAÇ ÇAĞRISI üretti:{R} {B}{cagri}{R}")


def tur(model, soru, araclarla: bool, sadece=None):
    """Tek bir soruyu baştan sona koştur. Döngüyü ekranda görünür kılar."""
    mesajlar = [{"role": "user", "content": soru}]
    print(f"{DIM}   kullanıcı:{R} {soru}\n")

    adim = 0
    while True:
        adim += 1
        # 3. bölümde tek araç veriyoruz: ekrandaki JSON sahnedekiyle
        # (theme.DEMO_CALL) birebir aynı çıksın.
        araclar = None
        if araclarla:
            araclar = ([t for t in TARIFLER if t["ad"] in sadece]
                       if sadece else TARIFLER)
        cevap = model.sor(mesajlar, araclar=araclar)

        bloklar = cevap.bloklar
        blok_yaz(bloklar)

        if not cevap.arac_bekliyor:
            return

        # --- KRİTİK: fonksiyonu MODEL DEĞİL, BU PROGRAM çalıştırıyor -------
        mesajlar.append(model.asistan_mesaji(cevap))
        sonuclar = []
        for b in bloklar:
            if b["tip"] != "cagri":
                continue
            girdi = b["girdi"]
            print(f"{YESIL}   >>> BİZİM PROGRAMIMIZ çalıştırıyor: "
                  f"{b['ad']}({', '.join(f'{k}={v!r}' for k, v in girdi.items())}){R}")
            sonuc = ARACLAR[b["ad"]](**girdi)
            print(f"{YESIL}   <<< sonuç: {sonuc}{R}")
            sonuclar.append({"ad": b["ad"], "icerik": sonuc})
        mesajlar += model.arac_sonuclari(sonuclar)
        print(f"{DIM}   (sonuç konuşmaya eklendi — döngü {adim}. turu tamamladı){R}\n")


def ollama_turu():
    """Aynı tarifi çok küçük bir modele ver: çağrıyı üretemediğini göster (S6)."""
    baslik("5", f"DAHA KÜÇÜK MODEL — {KUCUK_MODEL}", KIRMIZI)
    istem = (
        "Elindeki arac: kur_getir(para). Doviz kuru sorulursa SADECE su tek satiri "
        'yaz, baska hicbir sey yazma: {"tool": "kur_getir", "args": {"para": "USD"}}\n\n'
        f"Soru: {SORU}"
    )
    govde = json.dumps({"model": KUCUK_MODEL, "prompt": istem, "stream": False}).encode()
    try:
        req = urllib.request.Request(f"{SUNUCU}/api/generate", data=govde,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            cikti = json.loads(r.read())["response"].strip()
    except Exception as e:
        print(f"{KIRMIZI}   sunucuya ulaşılamadı: {e}{R}")
        return

    print(f"{DIM}   kullanıcı:{R} {SORU}")
    print(f"{KIRMIZI}   model:{R} {cikti[:400]}")
    try:
        json.loads(cikti)
        print(f"\n{YESIL}   ayrıştırıldı — bu sefer becerdi{R}")
    except json.JSONDecodeError:
        print(f"\n{KIRMIZI}{B}   >>> PARSE HATASI: bu bir araç çağrısı değil, "
              f"düz metin.{R}")
        print(f"{DIM}   Araç var, ama modelin onu kullanma alışkanlığı yok.")
        print(f"   İşte fine-tuning tam buraya giriyor.{R}")


def main():
    model = Model()

    print(f"\n{B}TOOL CALLING — canlı demo{R}   {DIM}model: {MODEL_ADI}{R}")

    # 1) Araçsız
    baslik("1", "ARAÇSIZ: model kendi ağırlıklarından cevaplasın", KIRMIZI)
    tur(model, SORU, araclarla=False)
    print(f"\n{KIRMIZI}   ^ kesim tarihinden sonrasını bilmiyor: ya tahmin "
          f"yürütüyor ya da bilmediğini söylüyor.{R}")
    duraklat()

    # 2) Tarif
    baslik("2", "ARACIN TARİFİ — modele verdiğimiz tek şey bu", MAVI)
    print(json.dumps(TARIFLER[0], indent=2, ensure_ascii=False))
    print(f"\n{DIM}   Model bu tarifi okur. Fonksiyonun kodunu GÖRMEZ.{R}")
    duraklat()

    # 3) Araçlı
    baslik("3", "ARAÇLI: aynı soru, aynı model", YESIL)
    tur(model, SORU, araclarla=True, sadece={"kur_getir"})
    print(f"\n{YESIL}   ^ aynı model, aynı ağırlıklar — tek fark: eli oldu.{R}")
    duraklat()

    # 4) İki tur dönen döngü
    baslik("4", "DÖNGÜ BİRDEN FAZLA TUR DÖNEBİLİR", MOR)
    tur(model, SORU2, araclarla=True)
    print(f"\n{MOR}   ^ model birden fazla araç çağırdı; her birini biz "
          f"çalıştırıp sonucu geri verdik.{R}")

    if OLLAMA_TURU:
        duraklat()
        ollama_turu()

    print(f"\n{B}Bitti.{R} {DIM}Kod: {REPO}{R}\n")


if __name__ == "__main__":
    main()
