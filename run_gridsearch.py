import pandas as pd
import numpy as np
import warnings
import time
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from segmentation_v1_2 import YadroSegmentation

warnings.filterwarnings('ignore')

class GridSearchYadroHPM:
    def __init__(self, min_samples=5, smote_ratio='auto', rf_estimators=100, cv_folds=10):
        self.min_samples = min_samples
        self.smote_ratio = smote_ratio
        self.rf_estimators = rf_estimators
        self.cv_folds = cv_folds

    def evaluate(self, X, y, eps, delta):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        yadro = YadroSegmentation(X_scaled, epsilon=eps, metric='euclidean')
        yadro.create_graph_with_weights()
        yadro.compute_density_variation_sequence()
        
        # O'zimiz delta ni beramiz
        core_pixels = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence, delta=delta, beta=self.min_samples)
        core_segments = yadro.partition_core_pixels(yadro.G, core_pixels, theta=0.1)
        segments, _ = yadro.expand_segments(yadro.G, yadro.Mt_sequence, core_segments)
        
        valid_indices = []
        for seg in segments:
            if len(seg) >= self.min_samples:
                valid_indices.extend(seg)
                
        valid_indices = sorted(valid_indices)
        mask = np.zeros(X.shape[0], dtype=bool)
        if len(valid_indices) > 0:
            mask[valid_indices] = True
        else:
            mask[:] = True # Fallback
            
        X_clean = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        y_clean = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
        
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
        try:
            X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
        except:
            return 0, 0 # if smote fails (not enough samples in a class)
            
        rf = RandomForestClassifier(n_estimators=self.rf_estimators, random_state=42)
        y_pred = cross_val_predict(rf, X_resampled, y_resampled, cv=self.cv_folds)
        
        acc = accuracy_score(y_resampled, y_pred)
        f1 = f1_score(y_resampled, y_pred, zero_division=0)
        return acc, f1

def run_grid_search(name, X, y, min_samples, smote_ratio):
    print(f"\n[{name}] bo'yicha GridSearch boshlandi...")
    eps_values = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    delta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    best_acc = 0
    best_f1 = 0
    best_params = {}
    
    start_time = time.time()
    
    gs = GridSearchYadroHPM(min_samples=min_samples, smote_ratio=smote_ratio)
    
    total = len(eps_values) * len(delta_values)
    i = 0
    for eps in eps_values:
        for d in delta_values:
            i += 1
            acc, f1 = gs.evaluate(X, y, eps, d)
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_params = {'eps': eps, 'delta': d}
            # print(f"  ({i}/{total}) eps={eps}, delta={d} -> Acc={acc*100:.2f}%, F1={f1*100:.2f}%")
            
    print(f"Topilgan eng yaxshi natija: Acc = {best_acc*100:.3f}%, F1 = {best_f1*100:.3f}%")
    print(f"Eng yaxshi parametrlar: eps = {best_params['eps']}, delta = {best_params['delta']}")
    print(f"Bajarilish vaqti: {time.time() - start_time:.2f} soniya")
    return best_acc, best_f1, best_params

def main():
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    print("                YadroSeg uchun GridSearch Optimizer                     ")
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")

    # Dataset 1
    df1 = pd.read_csv('datasets/dataset1_diabetes.csv')
    df1.dropna(subset=['glyhb'], inplace=True)
    df1['target'] = (df1['glyhb'] > 7.0).astype(int)
    features1 = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df1_clean = df1.dropna(subset=features1)
    
    run_grid_search("DATASET 1: Qandli Diabet", df1_clean[features1], df1_clean['target'], min_samples=5, smote_ratio=0.8)

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
    
    run_grid_search("DATASET 2: Gipertoniya", df2_clean[features2], df2_clean['target'], min_samples=5, smote_ratio=0.9)

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
    
    run_grid_search("DATASET 3: CKD", df3_clean[features3], df3_clean['target'], min_samples=3, smote_ratio=0.9)

if __name__ == "__main__":
    main()
