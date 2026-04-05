import json, os

downloads = os.path.join(os.environ['USERPROFILE'], 'Downloads')

# Read the baseline files that have per-language structure
for fname in ['B1_oracle.json', 'B2_whisper_auto.json', 'B3_static_mms.json', 'B4_sb_whisper.json']:
    fpath = os.path.join(downloads, fname)
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"{fname}")
    print(f"{'='*60}")
    
    # Print all data
    for key, val in data.items():
        print(f"  {key}: {val}")

# Also get A6 learned per-language
print(f"\n{'='*60}")
print(f"A6_LEARNED per-language CER")
print(f"{'='*60}")
with open(os.path.join(downloads, 'a6_learned_policy.json'), encoding='utf-8') as f:
    a6 = json.load(f)
for lang, info in a6['per_language'].items():
    print(f"  {lang}: CER={info['mean_cer']:.4f}, WER={info['mean_wer']:.4f}, LID={info['lid_accuracy']:.3f}, n={info['n_samples']}")

print(f"\n{'='*60}")
print(f"Step4 RULES per-language CER")
print(f"{'='*60}")
with open(os.path.join(downloads, 'eval_results.json'), encoding='utf-8') as f:
    s4 = json.load(f)
for lang, info in s4['per_language'].items():
    print(f"  {lang}: CER={info['mean_cer']:.4f}, WER={info['mean_wer']:.4f}, LID={info['lid_accuracy']:.3f}, n={info['n_samples']}")
