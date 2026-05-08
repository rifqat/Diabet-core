import pandas as pd
import io

with open('datasets/dataset3_ckd/Chronic_Kidney_Disease/chronic_kidney_disease.arff', 'r', encoding='utf-8') as f:
    lines = f.readlines()

data_lines = []
is_data = False
for line in lines:
    if line.strip().lower() == '@data':
        is_data = True
        continue
    if is_data and line.strip() and not line.startswith('%'):
        # split by comma, clean parts, and ensure we only keep exactly 25 columns
        parts = line.split(',')
        clean_parts = [p.strip().replace('\t', '') for p in parts]
        
        # some lines have a bunch of spaces creating empty fields at the end etc.
        # we know there are 25 features.
        
        # Remove empty string elements that might be trailing
        clean_parts_filtered = [p for p in clean_parts if p != '']
        
        # If we couldn't parse 25 correctly, just pad or truncate
        if len(clean_parts) > 25:
             # Take the first 25
             clean_parts = clean_parts[:25]
        elif len(clean_parts) < 25:
             # Pad with '?'
             clean_parts.extend(['?'] * (25 - len(clean_parts)))
             
        data_lines.append(','.join(clean_parts))

df_csv = pd.read_csv(io.StringIO('\n'.join(data_lines)), header=None)
df_csv.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

df_csv.to_csv('datasets/dataset3_ckd/Chronic_Kidney_Disease/ckd_clean.csv', index=False)
print("CKD clean.csv yaratildi!")
