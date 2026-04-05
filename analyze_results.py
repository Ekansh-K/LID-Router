"""Quick analysis of all result JSONs."""
import json, os

files = {
    'B1_oracle': 'results/B1_oracle.json',
    'B2_whisper_auto': 'results/B2_whisper_auto.json',
    'B3_static_mms': 'results/B3_static_mms.json',
    'B4_sb_whisper': 'results/B4_sb_whisper.json',
    'Step4_rules': 'results/eval_results.json',
    'A1_mms_lid_only': 'results/ablations/a1_mms_lid_only.json',
    'A6_rules': 'results/ablations/a6_rules_policy.json',
    'A6_learned': 'results/ablations/a6_learned_policy.json',
}

header = f"{'System':<22} {'CER':>8} {'WER':>8} {'LID_Acc':>8} {'LID_T3':>8} {'Route_A':>8} {'Route_B':>8} {'Route_C':>8} {'N':>6}"
print(header)
print('-' * len(header))

per_lang_data = {}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name:<22} FILE MISSING")
        continue
    with open(path) as f:
        d = json.load(f)
    cer = d.get('overall_mean_cer', 0)
    wer = d.get('overall_mean_wer', 0)
    lid = d.get('overall_lid_accuracy', 0)
    lid3 = d.get('overall_lid_top3_accuracy', 0)
    rd = d.get('routing_distribution', {})
    ra = rd.get('A', '-')
    rb = rd.get('B', '-')
    rc = rd.get('C', '-')
    n = d.get('n_samples', 0)
    print(f"{name:<22} {cer:>8.4f} {wer:>8.4f} {lid:>8.4f} {lid3:>8.4f} {str(ra):>8} {str(rb):>8} {str(rc):>8} {n:>6}")
    
    # Store per-language CER for key systems
    if name in ['B3_static_mms', 'Step4_rules', 'A6_learned', 'B1_oracle']:
        per_lang = d.get('per_language', {})
        per_lang_data[name] = {lang: info.get('mean_cer', 0) for lang, info in per_lang.items()}

# Per-language comparison
print("\n\n=== Per-Language CER Comparison (key systems) ===")
langs = sorted(set().union(*[set(v.keys()) for v in per_lang_data.values()]))
header2 = f"{'Lang':<8}"
for sysname in ['B1_oracle', 'B3_static_mms', 'Step4_rules', 'A6_learned']:
    if sysname in per_lang_data:
        header2 += f" {sysname:>16}"
header2 += "  Winner"
print(header2)
print('-' * len(header2))

wins = {'B3_static_mms': 0, 'Step4_rules': 0, 'A6_learned': 0}
for lang in langs:
    row = f"{lang:<8}"
    cers = {}
    for sysname in ['B1_oracle', 'B3_static_mms', 'Step4_rules', 'A6_learned']:
        if sysname in per_lang_data:
            val = per_lang_data[sysname].get(lang, float('inf'))
            cers[sysname] = val
            row += f" {val:>16.4f}"
    # Who wins (excluding oracle)?
    non_oracle = {k: v for k, v in cers.items() if k != 'B1_oracle'}
    if non_oracle:
        winner = min(non_oracle, key=non_oracle.get)
        row += f"  {winner}"
        if winner in wins:
            wins[winner] += 1
    print(row)

print(f"\n=== Language Wins (excluding oracle) ===")
for sys, count in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {sys}: {count} languages")

# Relative improvement of A6_learned over Step4_rules
if 'A6_learned' in per_lang_data and 'Step4_rules' in per_lang_data:
    a6_cer = sum(per_lang_data['A6_learned'].values()) / len(per_lang_data['A6_learned'])
    s4_cer = sum(per_lang_data['Step4_rules'].values()) / len(per_lang_data['Step4_rules'])
    b3_cer_avg = sum(per_lang_data['B3_static_mms'].values()) / len(per_lang_data['B3_static_mms']) if 'B3_static_mms' in per_lang_data else 0
    print(f"\n=== Summary ===")
    print(f"Step4 (rules) avg CER:  {s4_cer:.4f}")
    print(f"A6 (learned) avg CER:   {a6_cer:.4f}")
    print(f"B3 (static MMS) avg CER: {b3_cer_avg:.4f}")
    if s4_cer > 0:
        rel_imp = (s4_cer - a6_cer) / s4_cer * 100
        print(f"Relative improvement (learned vs rules): {rel_imp:.1f}%")
    if b3_cer_avg > 0:
        rel_vs_b3 = (b3_cer_avg - a6_cer) / b3_cer_avg * 100
        print(f"Relative improvement (learned vs B3):    {rel_vs_b3:.1f}%")
