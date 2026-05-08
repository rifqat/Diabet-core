import pandas as pd
import numpy as np
import time
import warnings
from scipy.io import arff
from hpm_model import HybridPredictionModel

warnings.filterwarnings('ignore')

print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
print("   Maqoladagi HPM (Gibrid Model)ni Haqiqiy Datasetlarda Sinash  ")
print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")

# =====================================================================
# DATASET 1: Diabetes Data (Virjiniya Universiteti)
# =====================================================================
print("\n[ DATASET 1: Qandli Diabet (Diabetes Data) ]")
try:
    df1 = pd.read_csv('datasets/dataset1_diabetes.csv')
    df1.dropna(subset=['glyhb'], inplace=True)
    df1['target'] = (df1['glyhb'] > 7.0).astype(int)
    features1 = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df1_clean = df1.dropna(subset=features1)
    
    X1 = df1_clean[features1]
    y1 = df1_clean['target']
    
    start_time = time.time()
    hpm1 = HybridPredictionModel(eps=1.5, min_samples=5, smote_ratio=0.8)
    hpm1.train_and_evaluate(X1, y1)
    end_time = time.time()
    print(f"-> Dataset 1 Muvaffaqiyatli tekshirildi. Bajarilish vaqti: {end_time - start_time:.4f} soniya\n")
except Exception as e:
    print(f"Dataset 1 bilan muammo chiqdi: {e}")


# =====================================================================
# DATASET 2: Men's Dataset (Gipertoniya)
# =====================================================================
print("\n[ DATASET 2: Gipertoniya (Men's Dataset) ]")
try:
    df2 = pd.read_csv('datasets/dataset2_hypertension.csv', sep=';')
    
    # Vergul muammosini hal qilish: barcha ustunlarni tekshiramiz
    for col in ['bmi', 'wc', 'hc', 'whr', 'SBP', 'DBP']:
        if col in df2.columns:
            # Type error bermasligi uchun string tipiga majburlab keyin tochkaga almashtiramiz
            df2[col] = df2[col].astype(str).str.replace(',', '.').astype(float)
            
    df2['target'] = (df2['SBP'] > 140).astype(int)
    
    if 'Is.Obese' in df2.columns:
        df2['is_obese'] = df2['Is.Obese'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    
    features2 = ['Age', 'is_obese', 'bmi', 'wc', 'hc', 'whr']
    df2_clean = df2.dropna(subset=features2)
    
    X2 = df2_clean[features2]
    y2 = df2_clean['target']
    
    start_time = time.time()
    hpm2 = HybridPredictionModel(eps=1.5, min_samples=5, smote_ratio=0.9)
    hpm2.train_and_evaluate(X2, y2)
    end_time = time.time()
    print(f"-> Dataset 2 Muvaffaqiyatli tekshirildi. Bajarilish vaqti: {end_time - start_time:.4f} soniya\n")
except Exception as e:
    print(f"Dataset 2 bilan muammo chiqdi: {e}")


# =====================================================================
# DATASET 3: Chronic Kidney Disease (Buyrak Kasalligi) 
# =====================================================================
print("\n[ DATASET 3: CKD (Qon bosimi va Diabet bog'liqligi) ]")
try:
    df3 = pd.read_csv('datasets/dataset3_ckd/Chronic_Kidney_Disease/ckd_clean.csv')
    
    df3['dm'] = df3['dm'].astype(str).str.replace('\t', '').str.strip()
    df3['htn'] = df3['htn'].astype(str).str.replace('\t', '').str.strip()
    
    # Missing values as nan
    df3.replace('?', np.nan, inplace=True)
    df3_clean = df3.dropna(subset=['age', 'bp', 'htn', 'dm'])
    
    # Faqat toza "yes" yoki "no" bo'lgan qatorlar
    valid_mask = df3_clean['htn'].isin(['yes', 'no']) & df3_clean['dm'].isin(['yes', 'no'])
    df3_clean = df3_clean[valid_mask]
    
    df3_clean['htn_num'] = (df3_clean['htn'] == 'yes').astype(float)
    df3_clean['target'] = (df3_clean['dm'] == 'yes').astype(int)
    
    # Cast to float
    df3_clean['age'] = df3_clean['age'].astype(float)
    df3_clean['bp'] = df3_clean['bp'].astype(float)
    
    features3 = ['age', 'bp', 'htn_num']
    X3 = df3_clean[features3]
    y3 = df3_clean['target']
    
    # MinPts va Eps pastroq qilinadi bu yerda (maqolaga ko'ra)
    start_time = time.time()
    hpm3 = HybridPredictionModel(eps=1.0, min_samples=3, smote_ratio=0.9)
    hpm3.train_and_evaluate(X3, y3)
    end_time = time.time()
    print(f"-> Dataset 3 Muvaffaqiyatli tekshirildi. Bajarilish vaqti: {end_time - start_time:.4f} soniya\n")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Dataset 3 bilan muammo chiqdi: {e}")

print("\n='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
print("                      Barcha testlar yakunlandi!")
print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
