import json, re
from jiwer import cer, wer
import numpy as np
from collections import defaultdict

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()

d = json.load(open('results/step8_track3_learned.json', encoding='utf-8'))
by_lang = defaultdict(lambda: {'r_cer':[], 'n_cer':[], 'r_wer':[], 'n_wer':[]})

for r in d['records']:
    ref, hyp = r.get('reference',''), r.get('hypothesis','')
    if not ref.strip():
        continue
    lang = r['true_lang']
    by_lang[lang]['r_cer'].append(r.get('cer', cer(ref, hyp)))
    by_lang[lang]['r_wer'].append(r.get('wer', wer(ref, hyp)))
    nref, nhyp = normalize(ref), normalize(hyp)
    if nref.strip():
        try:
            by_lang[lang]['n_cer'].append(cer(nref, nhyp))
            by_lang[lang]['n_wer'].append(wer(nref, nhyp))
        except Exception:
            pass
    else:
        by_lang[lang]['n_cer'].append(0.0)
        by_lang[lang]['n_wer'].append(0.0)

print("Step8 Learned — Normalized vs Raw CER/WER (lowercase + no punctuation)")
print("-" * 70)
fmt = "{:<6}  {:>9}  {:>9}  {:>7}  {:>9}  {:>9}  {:>7}"
print(fmt.format("Lang", "Raw CER", "Norm CER", "D_CER", "Raw WER", "Norm WER", "D_WER"))
print("-" * 70)

rc_all, nc_all, rw_all, nw_all = [], [], [], []
for lang in sorted(by_lang):
    v = by_lang[lang]
    rc, nc = np.mean(v['r_cer']), np.mean(v['n_cer'])
    rw, nw = np.mean(v['r_wer']), np.mean(v['n_wer'])
    rc_all += v['r_cer']; nc_all += v['n_cer']
    rw_all += v['r_wer']; nw_all += v['n_wer']
    delta_c = nc - rc
    delta_w = nw - rw
    print(fmt.format(lang, f"{rc:.4f}", f"{nc:.4f}", f"{delta_c:+.4f}", f"{rw:.4f}", f"{nw:.4f}", f"{delta_w:+.4f}"))

print("-" * 70)
print(fmt.format("MEAN", f"{np.mean(rc_all):.4f}", f"{np.mean(nc_all):.4f}", f"{np.mean(nc_all)-np.mean(rc_all):+.4f}", f"{np.mean(rw_all):.4f}", f"{np.mean(nw_all):.4f}", f"{np.mean(nw_all)-np.mean(rw_all):+.4f}"))
