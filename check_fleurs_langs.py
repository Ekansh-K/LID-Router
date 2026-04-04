#!/usr/bin/env python3
"""
Check which SUBSET_30_FLEURS languages can actually be loaded from FLEURS.
**Best run on Kaggle (which has compatible datasets version)**
Local check may fail due to datasets library version incompatibility.
"""
import sys
from evaluation.data_loader import SUBSET_30_FLEURS

print(f"SUBSET_30_FLEURS contains {len(SUBSET_30_FLEURS)} languages:\n")

# Group A: High-Resource
print("Group A (High-Resource):")
group_a = SUBSET_30_FLEURS[:8]
for i, lang in enumerate(group_a, 1):
    print(f"  {i}. {lang}")

# Group B: Confusion Pairs
print("\nGroup B (Confusion Pairs):")
group_b = SUBSET_30_FLEURS[8:18]
for i, lang in enumerate(group_b, 1):
    print(f"  {i}. {lang}")

# Group C: Low-Resource
print("\nGroup C (Low-Resource):")
group_c = SUBSET_30_FLEURS[18:30]
for i, lang in enumerate(group_c, 1):
    print(f"  {i}. {lang}")

print(f"\n{'='*60}")
print("TO CHECK AVAILABILITY ON KAGGLE:")
print("  Add this cell to Kaggle_Notebook.ipynb after Step 0 (env setup):\n")
print("""
from datasets import load_dataset
from evaluation.data_loader import SUBSET_30_FLEURS

available = []
failed = []

for lang_code in SUBSET_30_FLEURS:
    try:
        ds = load_dataset("google/fleurs", lang_code, split="validation", 
                         streaming=True, trust_remote_code=True)
        next(iter(ds))
        print(f"{lang_code}: OK")
        available.append(lang_code)
    except Exception as e:
        print(f"{lang_code}: FAILED")
        failed.append(lang_code)

print(f"\\nAvailable: {len(available)}/30")
print(f"Failed: {failed}")
""")
