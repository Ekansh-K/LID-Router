import json

with open('Kaggle_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    src = ''.join(cell.get('source', []))[:120].replace('\n', ' | ')
    if outputs:
        total_chars = 0
        for o in outputs:
            total_chars += len(''.join(o.get('text', [])))
            total_chars += len(''.join(o.get('data', {}).get('text/plain', [])))
        print(f'Cell {i}: {total_chars} output chars | src: {src}')
    else:
        if cell['cell_type'] == 'code':
            print(f'Cell {i}: NO OUTPUT | src: {src}')
