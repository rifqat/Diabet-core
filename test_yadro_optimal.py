import pandas as pd
import numpy as np
import warnings
import time
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from segmentation_v1_2 import YadroSegmentation

warnings.filterwarnings('ignore')

def run_optimal_yadro(name, X, y, eps, delta, min_samples, smote_ratio):
    print(f"\n[ {name} ustida optimal o'lchov... ]")
    start_time = time.time()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    yadro = YadroSegmentation(X_scaled, epsilon=eps, metric='euclidean')
    yadro.create_graph_with_weights()
    yadro.compute_density_variation_sequence()
    
    core_pixels = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence, delta=delta, beta=min_samples)
    core_segments = yadro.partition_core_pixels(yadro.G, core_pixels, theta=0.1)
    segments, _ = yadro.expand_segments(yadro.G, yadro.Mt_sequence, core_segments)
    
    valid_indices = []
    for seg in segments:
        if len(seg) >= min_samples:
            valid_indices.extend(seg)
            
    valid_indices = sorted(valid_indices)
    mask = np.zeros(X.shape[0], dtype=bool)
    if len(valid_indices) > 0:
        mask[valid_indices] = True
    else:
        mask[:] = True
        
    X_clean = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
    y_clean = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
    print(f"[{name}] Original: {X.shape[0]}, Cleaned: {X_clean.shape[0]}, Outliers: {X.shape[0] - X_clean.shape[0]}")
    
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
        
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred = cross_val_predict(rf, X_resampled, y_resampled, cv=10)
    
    end_time = time.time()
    print(f"-> {name} uchun ishlash vaqti: {end_time - start_time:.4f} soniya")

def main():
    # Dataset 1
    df1 = pd.read_csv('datasets/dataset1_diabetes.csv')
    df1.dropna(subset=['glyhb'], inplace=True)
    df1['target'] = (df1['glyhb'] > 7.0).astype(int)
    features1 = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df1_clean = df1.dropna(subset=features1)
    run_optimal_yadro("DATASET 1: Qandli Diabet", df1_clean[features1], df1_clean['target'], eps=0.85, delta=0.1, min_samples=5, smote_ratio=0.8)

    # Dataset 2
    df2 = pd.read_csv('datasets/dataset2_hypertension.csv', sep=';')
    for col in ['bmi', 'wc', 'hc', 'whr', 'SBP', 'DBP']:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.replace(',', '.').astype(float)
    df2['target'] = (df2['SBP'] > 140).astype(int)
    if 'Is.Obese' in df2.columns:
        df2['is_obese'] = df2['Is.Obese'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    features2 = ['Age', 'is_obese', 'bmi', 'wc', 'hc', 'whr']
    df2_clean = df2.dropna(subset=features2)
    run_optimal_yadro("DATASET 2: Gipertoniya", df2_clean[features2], df2_clean['target'], eps=0.8, delta=0.1, min_samples=5, smote_ratio=0.9)

    # Dataset 3
    df3 = pd.read_csv('datasets/dataset3_ckd/Chronic_Kidney_Disease/ckd_clean.csv')
    df3['dm'] = df3['dm'].astype(str).str.replace('\t', '').str.strip()
    df3['htn'] = df3['htn'].astype(str).str.replace('\t', '').str.strip()
    df3.replace('?', np.nan, inplace=True)
    df3_clean = df3.dropna(subset=['age', 'bp', 'htn', 'dm'])
    valid_mask = df3_clean['htn'].isin(['yes', 'no']) & df3_clean['dm'].isin(['yes', 'no'])
    df3_clean = df3_clean[valid_mask]
    df3_clean['htn_num'] = (df3_clean['htn'] == 'yes').astype(float)
    df3_clean['target'] = (df3_clean['dm'] == 'yes').astype(int)
    df3_clean['age'] = df3_clean['age'].astype(float)
    df3_clean['bp'] = df3_clean['bp'].astype(float)
    features3 = ['age', 'bp', 'htn_num']
    run_optimal_yadro("DATASET 3: CKD", df3_clean[features3], df3_clean['target'], eps=0.85, delta=0.1, min_samples=3, smote_ratio=0.9)

if __name__ == "__main__":
    main()
