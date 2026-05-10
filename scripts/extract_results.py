import json
import sys

# Try both notebooks
for nbfile in ['Kaggle_Notebook.ipynb', 'Kaggle_Notebook copy.ipynb']:
    try:
        with open(nbfile) as f:
            nb = json.load(f)
    except:
        continue
    
    print(f"\n{'='*60}")
    print(f"NOTEBOOK: {nbfile}")
    print(f"Total cells: {len(nb['cells'])}")
    print(f"{'='*60}")
    
    for i, cell in enumerate(nb['cells']):
        if cell.get('outputs'):
            for out in cell['outputs']:
                text = ''
                if 'text' in out:
                    text = ''.join(out['text'])
                elif 'data' in out and 'text/plain' in out['data']:
                    text = ''.join(out['data']['text/plain'])
                
                # Search for result-like content
                keywords = ['cer', 'wer', 'overall_mean', 'lid_accuracy', 
                           'routing_distribution', 'per_language', 'ablation',
                           'B1_oracle', 'B2_whisper', 'B3_static', 'B4_sb',
                           'a6_learned', 'a6_rules', 'a1_mms']
                if any(kw in text.lower() for kw in keywords):
                    print(f"\n--- Cell {i} (type={cell['cell_type']}) ---")
                    # Print source first
                    src = ''.join(cell.get('source', []))
                    if len(src) > 200:
                        src = src[:200] + '...'
                    print(f"SOURCE: {src}")
                    print(f"OUTPUT ({len(text)} chars):")
                    if len(text) > 1000:
                        print(text[:1000] + '...[truncated]')
                    else:
                        print(text)

# Also check result files - maybe some are actual results
print("\n\n" + "="*60)
print("CHECKING RESULT FILES")
print("="*60)
import os
for root, dirs, files in os.walk('results'):
    for fname in files:
        fpath = os.path.join(root, fname)
        with open(fpath) as f:
            content = f.read(200)
        if content.startswith('{"metadata"'):
            print(f"  {fpath}: IS A NOTEBOOK (not results)")
        else:
            print(f"  {fpath}: ACTUAL JSON - first 200 chars: {content[:200]}")
