import pandas as pd
import numpy as np
import time
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from segmentation_v1_2 import YadroSegmentation

warnings.filterwarnings('ignore')

class YadroHybridPredictionModel:
    def __init__(self, eps=0.5, min_samples=5, smote_ratio='auto', rf_estimators=100, cv_folds=10, use_auto_eps=False):
        self.eps = eps
        self.min_samples = min_samples
        self.smote_ratio = smote_ratio
        self.rf_estimators = rf_estimators
        self.cv_folds = cv_folds
        self.use_auto_eps = use_auto_eps
        
    def train_and_evaluate(self, X, y):
        print(f"[1] Dastlabki ma'lumotlar hajmi: {X.shape}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # YadroSegmentation yordamida clustering
        yadro = YadroSegmentation(X_scaled, epsilon=self.eps, metric='euclidean')
        if self.use_auto_eps:
            yadro.auto_select_epsilon(k=5, min_cluster_ratio=0.08)
            print(f" -> Auto epsilon tanlandi: {yadro.epsilon:.4f}")
            
        yadro.create_graph_with_weights()
        yadro.compute_density_variation_sequence()
        optimal_delta, _ = yadro.find_optimal_params_elbow(beta=self.min_samples, visualize_elbow=False)
        core_pixels = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence, delta=optimal_delta, beta=self.min_samples)
        core_segments = yadro.partition_core_pixels(yadro.G, core_pixels, theta=0.1)
        segments, low_confidence_pixels = yadro.expand_segments(yadro.G, yadro.Mt_sequence, core_segments)
        
        valid_indices = []
        for seg in segments:
            if len(seg) >= self.min_samples:
                valid_indices.extend(seg)
                
        valid_indices = sorted(valid_indices)
        mask = np.zeros(X.shape[0], dtype=bool)
        if len(valid_indices) > 0:
            mask[valid_indices] = True
        else:
            print(" -> Barcha nuqtalar tasnifdan tashqarida qoldi (outlier). DBSCAN o'rniga barchasi olinadi.")
            mask[:] = True
            
        X_clean = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        y_clean = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
        
        print(f"[2] YadroSeg tozalagandan keyingi hajm: {X_clean.shape}")
        
        # SMOTE
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
        
        print(f"[3] SMOTE tenglashtirgandan keyingi hajm: {X_resampled.shape}")
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=self.rf_estimators, random_state=42)
        print(f"[4] Random Forest modeli {self.cv_folds}-Fold bo'yicha o'rganmoqda va baholamoqda...")
        y_pred = cross_val_predict(rf, X_resampled, y_resampled, cv=self.cv_folds)
        
        tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()
        accuracy = accuracy_score(y_resampled, y_pred)
        precision = precision_score(y_resampled, y_pred, zero_division=0)
        recall = recall_score(y_resampled, y_pred, zero_division=0)
        f1 = f1_score(y_resampled, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\n=== Modelni Baholash Natijalari (YadroSeg) ===")
        print(f"Aniqligi (Accuracy):    {accuracy * 100:.3f} %")
        print(f"Ta'sirchanlik (Recall): {recall * 100:.3f} %")
        print(f"Maxsusligi (Specificity): {specificity * 100:.3f} %")
        print(f"F1 Qiymati (F1 Score):  {f1 * 100:.3f} %")
        print(f"Keng qamrov (Precision):{precision * 100:.3f} %")
        
        rf.fit(X_resampled, y_resampled)
        return rf

def main():
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    print("   YadroSeg HPM (Gibrid Model)ni Haqiqiy Datasetlarda Sinash  ")
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")

    # Dataset 1
    print("\n[ DATASET 1: Qandli Diabet (Diabetes Data) ]")
    df1 = pd.read_csv('datasets/dataset1_diabetes.csv')
    df1.dropna(subset=['glyhb'], inplace=True)
    df1['target'] = (df1['glyhb'] > 7.0).astype(int)
    features1 = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df1_clean = df1.dropna(subset=features1)
    
    # DBSCAN hpm_model.py da eps=1.5, min_samples=5 deb berilgan.
    # YadroSeg da auto_eps ishlatsak bo'ladi.
    start_time = time.time()
    hpm1 = YadroHybridPredictionModel(eps=0.8, min_samples=5, smote_ratio=0.8, use_auto_eps=True)
    hpm1.train_and_evaluate(df1_clean[features1], df1_clean['target'])
    end_time = time.time()
    print(f"-> Dataset 1 Bajarilish vaqti (YadroSeg): {end_time - start_time:.4f} soniya\n")

    # Dataset 2
    print("\n[ DATASET 2: Gipertoniya (Men's Dataset) ]")
    df2 = pd.read_csv('datasets/dataset2_hypertension.csv', sep=';')
    for col in ['bmi', 'wc', 'hc', 'whr', 'SBP', 'DBP']:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.replace(',', '.').astype(float)
    df2['target'] = (df2['SBP'] > 140).astype(int)
    if 'Is.Obese' in df2.columns:
        df2['is_obese'] = df2['Is.Obese'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    features2 = ['Age', 'is_obese', 'bmi', 'wc', 'hc', 'whr']
    df2_clean = df2.dropna(subset=features2)
    
    start_time = time.time()
    hpm2 = YadroHybridPredictionModel(eps=0.8, min_samples=5, smote_ratio=0.9, use_auto_eps=True)
    hpm2.train_and_evaluate(df2_clean[features2], df2_clean['target'])
    end_time = time.time()
    print(f"-> Dataset 2 Bajarilish vaqti (YadroSeg): {end_time - start_time:.4f} soniya\n")

    # Dataset 3
    print("\n[ DATASET 3: CKD (Qon bosimi va Diabet bog'liqligi) ]")
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
    
    start_time = time.time()
    hpm3 = YadroHybridPredictionModel(eps=0.8, min_samples=3, smote_ratio=0.9, use_auto_eps=True)
    hpm3.train_and_evaluate(df3_clean[features3], df3_clean['target'])
    end_time = time.time()
    print(f"-> Dataset 3 Bajarilish vaqti (YadroSeg): {end_time - start_time:.4f} soniya\n")

if __name__ == "__main__":
    main()
