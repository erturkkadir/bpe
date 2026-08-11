#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys
import time

import torch
from torch.utils.data import DataLoader

TEMEL = "Qwen/Qwen2.5-0.5B-Instruct"
VERI = pathlib.Path("veri/ornekler.jsonl")
CIKTI = pathlib.Path("adapter")

SISTEM = (
    "Elindeki araclar:\n"
    '  kur_getir(para)  -> bir para biriminin guncel TL karsiligi\n'
    "  saat()           -> su anki tarih ve saat\n"
    "Bu araclardan biri gerekiyorsa SADECE tek satirlik cagriyi yaz, "
    "baska hicbir sey yazma."
)
SORU = "Bugün 1 dolar kaç TL?"

R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
MAVI = "\033[38;5;75m"; YESIL = "\033[38;5;79m"; KIRMIZI = "\033[38;5;203m"
AMBER = "\033[38;5;215m"; MOR = "\033[38;5;141m"


def baslik(n, metin, renk=MAVI):
    print(f"\n{renk}{B}{'━' * 72}{R}")
    print(f"{renk}{B}  {n}. {metin}{R}")
    print(f"{renk}{B}{'━' * 72}{R}\n")


def duraklat(hizli, mesaj="devam için Enter"):
    if hizli:
        return
    try:
        input(f"\n{DIM}   [{mesaj}]{R}")
    except EOFError:
        pass


# ---- Araçlar (tool_demo.py ile aynı) ------------------------------------
def kur_getir(para: str) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"https://open.er-api.com/v6/latest/{para.upper()}", timeout=8) as r:
            return f"1 {para.upper()} = {json.loads(r.read())['rates']['TRY']:.2f} TL"
    except Exception:
        return f"1 {para.upper()} = 41.83 TL  (YEDEK SABİT DEĞER — servis kapalı)"


def saat() -> str:
    from datetime import datetime
    return datetime.now().strftime("%d.%m.%Y %H:%M")


ARACLAR = {"kur_getir": kur_getir, "saat": saat}


def cevapla(model, tok, soru, max_yeni=64):
    mesajlar = [{"role": "system", "content": SISTEM},
                {"role": "user", "content": soru}]
    metin = tok.apply_chat_template(mesajlar, tokenize=False,
                                    add_generation_prompt=True)
    girdi = tok(metin, return_tensors="pt").to(model.device)
    with torch.no_grad():
        cikti = model.generate(**girdi, max_new_tokens=max_yeni,
                               do_sample=False,
                               pad_token_id=tok.eos_token_id)
    return tok.decode(cikti[0][girdi["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def dene_ayristir(cikti):
    """Programımızın yapacağı şey: çıktıdan araç çağrısını ayrıştırmaya çalış."""
    try:
        veri = json.loads(cikti)
        if isinstance(veri, dict) and "tool" in veri:
            return veri
    except json.JSONDecodeError:
        pass
    return None


def rapor_et(cikti, basarili_renk=YESIL):
    print(f"   model: {cikti[:300]}")
    cagri = dene_ayristir(cikti)
    if cagri:
        print(f"\n{basarili_renk}{B}   >>> AYRIŞTIRILDI: {cagri}{R}")
        sonuc = ARACLAR[cagri["tool"]](**cagri.get("args", {}))
        print(f"{basarili_renk}   >>> fonksiyon çalıştı: {sonuc}{R}")
    else:
        print(f"\n{KIRMIZI}{B}   >>> PARSE HATASI — bu bir araç çağrısı değil.{R}")
        print(f"{DIM}   Araç var, alışkanlık yok.{R}")
    return cagri is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--epok", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    hizli = args.hizli

    if not VERI.exists():
        print(f"{KIRMIZI}{VERI} yok. Önce:  ./venv/bin/python veri_uret.py{R}")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{B}FINE-TUNING — canlı demo{R}   {DIM}model: {TEMEL} · cihaz: {cihaz}{R}")

    tok = AutoTokenizer.from_pretrained(TEMEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEMEL, dtype=torch.bfloat16 if cihaz == "cuda" else torch.float32).to(cihaz)

    # ---- 1) ÖNCE -------------------------------------------------------
    baslik("1", "ÖNCE — ham model, araç tarifi verilmiş hâlde", KIRMIZI)
    print(f"{DIM}   sistem:{R} {SISTEM}\n")
    print(f"{DIM}   kullanıcı:{R} {SORU}")
    once = cevapla(model, tok, SORU)
    rapor_et(once)
    duraklat(hizli)

    # ---- 2) VERİ -------------------------------------------------------
    baslik("2", "VERİ SETİ — (istek → ideal cevap) çiftleri", MOR)
    satirlar = [json.loads(s) for s in VERI.read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"{DIM}   dosya: {VERI} · {len(satirlar)} satır{R}")
    print(f"{AMBER}   üreten (öğretmen) model: claude-opus-5{R}   "
          f"{DIM}<- kapanışta buraya döneceğiz{R}\n")
    for s in satirlar[:3]:
        print(f"   {s['soru']}")
        print(f"{AMBER}      -> {s['cevap']}{R}")
    duraklat(hizli)

    # ---- 3) LoRA -------------------------------------------------------
    baslik("3", "LoRA — dev matrisi dondur, yanına ince ek tak", MOR)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    egitilen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    toplam = sum(p.numel() for p in model.parameters())
    print(f"   toplam parametre   : {toplam:,}")
    print(f"{YESIL}   eğitilen parametre : {egitilen:,}  "
          f"(%{100 * egitilen / toplam:.3f}){R}")
    duraklat(hizli)

    # ---- 4) EĞİTİM -----------------------------------------------------
    baslik("4", f"EĞİTİM — ileri geç, loss'u ölç, geriye yay, güncelle "
                f"({args.epok} tur)", AMBER)

    def ornek_hazirla(s):
        """Sadece CEVAP tokenlarından öğren (soru maskelenir)."""
        istem = tok.apply_chat_template(
            [{"role": "system", "content": SISTEM},
             {"role": "user", "content": s["soru"]}],
            tokenize=False, add_generation_prompt=True)
        tam = istem + s["cevap"] + tok.eos_token
        i_ids = tok(istem, add_special_tokens=False)["input_ids"]
        t_ids = tok(tam, add_special_tokens=False)["input_ids"]
        etiket = [-100] * len(i_ids) + t_ids[len(i_ids):]
        return {"input_ids": t_ids, "labels": etiket}

    veri = [ornek_hazirla(s) for s in satirlar]

    def topla(grup):
        n = max(len(x["input_ids"]) for x in grup)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in grup])
        lab = torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in grup])
        att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in grup])
        return ids.to(cihaz), lab.to(cihaz), att.to(cihaz)

    yukleyici = DataLoader(veri, batch_size=args.batch, shuffle=True, collate_fn=topla)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    t0, adim = time.time(), 0
    for epok in range(args.epok):
        for ids, lab, att in yukleyici:
            cikti = model(input_ids=ids, attention_mask=att, labels=lab)  # ileri geç
            cikti.loss.backward()                                          # geriye yay
            opt.step()                                                     # güncelle
            opt.zero_grad()
            adim += 1
            if adim % 5 == 0 or adim == 1:
                bar = "█" * max(int(cikti.loss.item() * 6), 1)
                print(f"   tur {epok + 1}  adım {adim:3d}   loss "
                      f"{cikti.loss.item():.4f}  {AMBER}{bar}{R}")
    sure = time.time() - t0
    model.eval()
    print(f"\n{YESIL}   eğitim bitti — {sure:.0f} saniye, {adim} adım{R}")
    duraklat(hizli)

    # ---- 5) SONRA ------------------------------------------------------
    baslik("5", "SONRA — aynı model, aynı soru, takılı ek ile", YESIL)
    print(f"{DIM}   kullanıcı:{R} {SORU}")
    sonra = cevapla(model, tok, SORU)
    ok = rapor_et(sonra)

    print(f"\n{DIM}   başka sorularla da deneyelim:{R}")
    for s in ["Euro kaça gidiyor?", "saat kaç acaba", "Sterlin ne alemde?"]:
        c = cevapla(model, tok, s)
        isaret = f"{YESIL}✓{R}" if dene_ayristir(c) else f"{KIRMIZI}✗{R}"
        print(f"   {isaret} {s:24s} -> {c[:70]}")

    # ---- 6) ÖZET -------------------------------------------------------
    model.save_pretrained(CIKTI)
    mb = sum(f.stat().st_size for f in CIKTI.rglob("*") if f.is_file()) / 1e6
    baslik("6", "ÖZET", MAVI)
    print(f"   {'ÖNCE':7s}: {once[:60]}")
    print(f"   {'SONRA':7s}: {sonra[:60]}")
    print(f"\n   eğitilen parametre : %{100 * egitilen / toplam:.3f}")
    print(f"   eğitim süresi      : {sure:.0f} sn")
    print(f"   ek (adapter) boyutu: {mb:.1f} MB   {DIM}(ana model ~1000 MB){R}")
    print(f"\n{B}   Modele yeni bilgi öğretmedik — nasıl davranacağını öğrettik.{R}\n")
    if not ok:
        print(f"{KIRMIZI}   (not: bu turda ayrıştırma başarısız — --epok değerini "
              f"artırıp tekrar dene){R}\n")


if __name__ == "__main__":
    main()
