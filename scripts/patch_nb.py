with open('results_analysis.ipynb', encoding='utf8') as f:
    text = f.read()

# Replace block 1
old1 = '''    "r_s8r   = load_json(RESULTS / 'step8_track3_rules.json')   # Step8 + Track3 rules\\n",
    "r_s8l   = load_json(RESULTS / 'step8_track3_learned.json') # Step8 + Track3 learned (FINAL)\\n",'''
new1 = '''    "r_s8r   = load_json(RESULTS / 'step8_track3_rules.json')   # Step8 + Track3 rules\\n",
    "r_s8l   = load_json(RESULTS / 'step8_track3_learned.json') # Step8 + Track3 learned\\n",
    "r_s9l   = load_json(RESULTS / 'step9_phase12_learned.json') # Step9 (Phase 1+2)\\n",
    "r_s10f  = load_json(RESULTS / 'step10_phase3_f0.json')      # Step10 (Phase 3 F0)\\n",'''
text = text.replace(old1, new1)

# Replace block 2
old2 = '''    "    ('Step8 Rules+Track3',     r_s8r,   'Final')\\n",
    "    ('Step8 Learned+Track3 ⭐', r_s8l,  'Final')\\n",'''
new2 = '''    "    ('Step8 Rules',            r_s8r,   'Ablation')\\n",
    "    ('Step8 Learned',          r_s8l,   'Ablation')\\n",
    "    ('Step9 Learned (Ph 1+2)', r_s9l,   'Final')\\n",
    "    ('Step10 F0 (Ph 3) ⭐',     r_s10f,  'Final')\\n",'''
text = text.replace(old2, new2)

# Replace block 3
old3 = '''    "    ('A6 Learned (old)',        r_a6),\\n",
    "    ('Step8 Learned+Track3 ⭐', r_s8l),\\n",
    "    ('Step8 Rules+Track3',     r_s8r),\\n",'''
new3 = '''    "    ('A6 Learned (old)',       r_a6),\\n",
    "    ('Step8 Learned',          r_s8l),\\n",
    "    ('Step9 Learned (Ph 1+2)', r_s9l),\\n",
    "    ('Step10 F0 (Ph 3) ⭐',     r_s10f),\\n",'''
text = text.replace(old3, new3)

# Replace block 4
old4 = '''    "    ('A6 Learned (old)',              r_a6),\\n",
    "    ('Step8 Rules+Track3',            r_s8r),\\n",
    "    ('Step8 Learned+Track3 ⭐',       r_s8l),\\n",'''
new4 = '''    "    ('A6 Learned (old)',              r_a6),\\n",
    "    ('Step8 Learned',                 r_s8l),\\n",
    "    ('Step9 Learned (Ph 1+2)',        r_s9l),\\n",
    "    ('Step10 F0 (Ph 3) ⭐',            r_s10f),\\n",'''
text = text.replace(old4, new4)

with open('results_analysis.ipynb', 'w', encoding='utf8') as f:
    f.write(text)

print('Patched successfully!')
