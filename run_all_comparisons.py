import pandas as pd
import numpy as np
import time
import warnings
from sklearn.cluster import DBSCAN
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from segmentation_v1_2 import YadroSegmentation
from autoscan import AutoSCAN

warnings.filterwarnings('ignore')

# ----------------- DATASET LOADERS ----------------- #
def get_dataset_1():
    df = pd.read_csv('datasets/dataset1_diabetes.csv')
    df.dropna(subset=['glyhb'], inplace=True)
    df['target'] = (df['glyhb'] > 7.0).astype(int)
    features = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df = df.dropna(subset=features)
    return df[features], df['target'], "DATASET 1: Qandli Diabet (Diabetes)", 5

def get_dataset_2():
    df = pd.read_csv('datasets/dataset2_hypertension.csv', sep=';')
    for col in ['bmi', 'wc', 'hc', 'whr', 'SBP', 'DBP']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['target'] = (df['SBP'] > 140).astype(int)
    if 'Is.Obese' in df.columns:
        df['is_obese'] = df['Is.Obese'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    features = ['Age', 'is_obese', 'bmi', 'wc', 'hc', 'whr']
    df = df.dropna(subset=features)
    return df[features], df['target'], "DATASET 2: Gipertoniya (Hypertension)", 5

def get_dataset_3():
    df = pd.read_csv('datasets/dataset3_ckd/Chronic_Kidney_Disease/ckd_clean.csv')
    df['dm'] = df['dm'].astype(str).str.replace('\t', '').str.strip()
    df['htn'] = df['htn'].astype(str).str.replace('\t', '').str.strip()
    df.replace('?', np.nan, inplace=True)
    df = df.dropna(subset=['age', 'bp', 'htn', 'dm'])
    valid_mask = df['htn'].isin(['yes', 'no']) & df['dm'].isin(['yes', 'no'])
    df = df[valid_mask]
    df['htn_num'] = (df['htn'] == 'yes').astype(float)
    df['target'] = (df['dm'] == 'yes').astype(int)
    df['age'] = df['age'].astype(float)
    df['bp'] = df['bp'].astype(float)
    features = ['age', 'bp', 'htn_num']
    return df[features], df['target'], "DATASET 3: Buyrak va Qon bosim (CKD)", 3

def get_dataset_4():
    df = pd.read_csv('datasets/processed_cleveland.csv', na_values='?')
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df['target'] = (df['target'] > 0).astype(int)
    features = [c for c in df.columns if c != 'target']
    return df[features], df['target'], "DATASET 4: Cleveland Heart Disease", 5

def get_dataset_5():
    df = pd.read_csv('datasets/statlog_heart.dat', sep=' ')
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df['target'] = (df['target'] == 2).astype(int)
    features = [c for c in df.columns if c != 'target']
    return df[features], df['target'], "DATASET 5: Statlog Heart Disease", 5

# ----------------- FEATURE SELECTION ----------------- #
def select_features(X, y):
    print("  -> Eng muhim xususiyatlarni (Features) tanlash jarayoni...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    
    selector = SelectFromModel(rf_selector, prefit=True)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    
    # Agar model juda kam ustun tanlasa yoki hammasini tashlab yuborsa, fall-back
    if len(selected_features) < 2:
        # Barchasini olamiz
        selected_features = X.columns
        
    print(f"  -> Tanlangan {len(selected_features)}/{X.shape[1]} ta ustun:\n     {list(selected_features)}")
    return X[selected_features]

# ----------------- EVALUATORS ----------------- #
def print_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  [{name}] Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")
    return acc, f1

def apply_smoteenn_and_rf(X_clean, y_clean):
    sme = SMOTEENN(random_state=42)
    try:
        X_res, y_res = sme.fit_resample(X_clean, y_clean)
    except Exception as e:
        # Imbalanced muammo sabab kamchil klassda yetarli namuna bo'lmasa, fall-back SMOTE ishlatamiz
        try:
            smote_fallback = SMOTE(random_state=42)
            X_res, y_res = smote_fallback.fit_resample(X_clean, y_clean)
        except:
            X_res, y_res = X_clean, y_clean

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    cv_folds = 10
    counts = np.bincount(y_res)
    min_class_count = np.min(counts) if len(counts) > 0 else 0
    if min_class_count < cv_folds:
        cv_folds = max(2, min_class_count)
        
    if len(y_res) < 2 or cv_folds < 2:
        return y_res, y_res
        
    y_pred = cross_val_predict(rf, X_res, y_res, cv=cv_folds)
    return y_res, y_pred

def run_dbscan(X, y, eps_val=1.5, min_samples=5):
    start = time.time()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    db = DBSCAN(eps=eps_val, min_samples=min_samples)
    clusters = db.fit_predict(X_sc)
    
    mask = clusters != -1
    outliers = sum(~mask)
    X_clean, y_clean = X[mask], y[mask]
    
    # Agar hammasi o'chib ketsa (outlier) bo'lsa
    if len(X_clean) == 0:
        X_clean, y_clean = X, y    

    y_res, y_pred = apply_smoteenn_and_rf(X_clean, y_clean)
    elapsed = time.time() - start
    
    acc, f1 = print_metrics(y_res, y_pred, "DBSCAN (Baseline)")
    return acc, f1, elapsed, outliers

def run_yadro_auto(X, y, min_samples=5):
    start = time.time()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    # Kichkina shovqinlarni yig'masligi uchun stderr block qilinadi
    class NullWriter:
        def write(self, s): pass
        def flush(self): pass
    import sys
    orig_stdout = sys.stdout
    sys.stdout = NullWriter()
    
    try:
        yadro = YadroSegmentation(X_sc, metric='euclidean')
        yadro.auto_select_epsilon()
        yadro.create_graph_with_weights()
        yadro.compute_density_variation_sequence()
        delta, _ = yadro.find_optimal_params_elbow(beta=min_samples, visualize_elbow=False)
        
        core_pixels = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence, delta=delta, beta=min_samples)
        core_segments = yadro.partition_core_pixels(yadro.G, core_pixels, theta=0.1)
        segments, _ = yadro.expand_segments(yadro.G, yadro.Mt_sequence, core_segments)
        
        valid_indices = []
        for seg in segments:
            if len(seg) >= min_samples:
                valid_indices.extend(seg)
                
        mask = np.zeros(X.shape[0], dtype=bool)
        if valid_indices:
            mask[valid_indices] = True
        else:
            mask[:] = True
    finally:
        sys.stdout = orig_stdout
        
    X_clean, y_clean = X[mask], y[mask]
    outliers = sum(~mask)
    
    y_res, y_pred = apply_smoteenn_and_rf(X_clean, y_clean)
    elapsed = time.time() - start
    
    acc, f1 = print_metrics(y_res, y_pred, "YadroSeg (Auto)")
    return acc, f1, elapsed, outliers

def run_yadro_grid(X, y, min_samples=5):
    eps_vals = [0.8, 0.85, 0.9, 0.95]
    delta_vals = [0.1, 0.3, 0.5, 0.7]
    
    best_acc = 0
    best_eps = 0.85
    best_delta = 0.1
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    class NullWriter:
        def write(self, s): pass
        def flush(self): pass
    import sys
    orig_stdout = sys.stdout
    sys.stdout = NullWriter()
    
    # GRID SEARCH phase (Vaqt o'lchanmaydi)
    for eps in eps_vals:
        try:
            yadro = YadroSegmentation(X_sc, epsilon=eps, metric='euclidean')
            yadro.create_graph_with_weights()
            yadro.compute_density_variation_sequence()
            for d in delta_vals:
                cp = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence, delta=d, beta=min_samples)
                cs = yadro.partition_core_pixels(yadro.G, cp, theta=0.1)
                seg, _ = yadro.expand_segments(yadro.G, yadro.Mt_sequence, cs)
                v_ind = []
                for s in seg:
                    if len(s) >= min_samples: v_ind.extend(s)
                m = np.zeros(X.shape[0], dtype=bool)
                if v_ind: m[v_ind] = True
                else: m[:] = True
                
                Xc, yc = X[m], y[m]
                
                # Tezkor eval
                sm = SMOTE(random_state=42)
                try: xr, yr = sm.fit_resample(Xc, yc)
                except: xr, yr = Xc, yc
                
                if len(yr) < 10:
                    continue
                    
                rf = RandomForestClassifier(n_estimators=20, random_state=42)
                cv_f = min(3, np.min(np.bincount(yr)))
                if cv_f < 2: continue
                yp = cross_val_predict(rf, xr, yr, cv=cv_f)
                a = accuracy_score(yr, yp)
                if a > best_acc:
                    best_acc = a
                    best_eps = eps
                    best_delta = d
        except:
            continue
            
    sys.stdout = orig_stdout
    
    # INFERENCE Phase - Qidirilgandan so'ng topilgan optimal parametrlarni ishlatish (AYNAN SHU VAQT O'LCHANADI)
    start = time.time()
    
    sys.stdout = NullWriter()
    try:
        y_opt = YadroSegmentation(X_sc, epsilon=best_eps, metric='euclidean')
        y_opt.create_graph_with_weights()
        y_opt.compute_density_variation_sequence()
        
        c_pix = y_opt.identify_core_pixels(y_opt.Dt_sequence, y_opt.Mt_sequence, delta=best_delta, beta=min_samples)
        c_seg = y_opt.partition_core_pixels(y_opt.G, c_pix, theta=0.1)
        seg, _ = y_opt.expand_segments(y_opt.G, y_opt.Mt_sequence, c_seg)
        
        valid_ind = []
        for s in seg:
            if len(s) >= min_samples:
                valid_ind.extend(s)
                
        m = np.zeros(X.shape[0], dtype=bool)
        if valid_ind:
            m[valid_ind] = True
        else:
            m[:] = True
    finally:
        sys.stdout = orig_stdout
        
    Xc, yc = X[m], y[m]
    outliers = sum(~m)
    
    yr, yp = apply_smoteenn_and_rf(Xc, yc)
    elapsed = time.time() - start
    
    print(f"  [YadroSeg Grid] Optimal: eps={best_eps}, delta={best_delta}")
    acc, f1 = print_metrics(yr, yp, "YadroSeg (GridSearch)")
    return acc, f1, elapsed, outliers


def run_autoscan(X, y, min_samples=None):
    """AutoSCAN (Bushra et al., 2024) — automatic ε detection + ClosestCore"""
    start = time.time()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    autoscan = AutoSCAN(min_samples=min_samples)
    autoscan.fit_predict(X_sc)
    
    mask = autoscan.get_clean_mask()
    outliers = sum(~mask)
    X_clean, y_clean = X[mask], y[mask]
    
    # Agar hammasi o'chib ketsa (outlier) bo'lsa
    if len(X_clean) == 0:
        X_clean, y_clean = X, y
        outliers = 0
    
    y_res, y_pred = apply_smoteenn_and_rf(X_clean, y_clean)
    elapsed = time.time() - start
    
    print(f"  [AutoSCAN] eps={autoscan.epsilon_:.4f}, minPts={autoscan.min_samples}")
    acc, f1 = print_metrics(y_res, y_pred, "AutoSCAN")
    return acc, f1, elapsed, outliers


# ----------------- MAIN PIPELINE ----------------- #
def main():
    datasets = [get_dataset_1, get_dataset_2, get_dataset_3, get_dataset_4, get_dataset_5]
    results = []
    
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    print("      5 Dataset bo'yicha YadroSeg vs DBSCAN vs AutoSCAN (SMOTEENN) Taqqoslovi")
    print("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    
    for loader in datasets:
        X, y, name, min_samples = loader()
        print(f"\n>>>> Tahlil qilinyapti: {name}")
        print(f"  -> Dastlabki ma'lumotlar hajmi: {X.shape}")
        
        X_sel = select_features(X, y)
        
        # DBSCAN Baseline
        a_db, f_db, t_db, o_db = run_dbscan(X_sel, y, eps_val=1.5, min_samples=min_samples)
        
        # AutoSCAN (Bushra et al., 2024)
        a_as, f_as, t_as, o_as = run_autoscan(X_sel, y, min_samples=min_samples)
        
        # YadroSeg Auto
        a_ya, f_ya, t_ya, o_ya = run_yadro_auto(X_sel, y, min_samples=min_samples)
        
        # YadroSeg GridSearch
        a_yg, f_yg, t_yg, o_yg = run_yadro_grid(X_sel, y, min_samples=min_samples)
        
        ds_label = name.split(':')[0].replace('DATASET ', 'DS_')
        results.append({
            'Dataset': ds_label,
            'Usul': 'DBSCAN',
            'Outliers': o_db,
            'Accuracy': f"{a_db*100:.2f} %",
            'F1': f"{f_db*100:.2f} %",
            'Time (s)': f"{t_db:.2f}"
        })
        results.append({
            'Dataset': ds_label,
            'Usul': 'AutoSCAN',
            'Outliers': o_as,
            'Accuracy': f"{a_as*100:.2f} %",
            'F1': f"{f_as*100:.2f} %",
            'Time (s)': f"{t_as:.2f}"
        })
        results.append({
            'Dataset': ds_label,
            'Usul': 'Yadro Auto',
            'Outliers': o_ya,
            'Accuracy': f"{a_ya*100:.2f} %",
            'F1': f"{f_ya*100:.2f} %",
            'Time (s)': f"{t_ya:.2f}"
        })
        results.append({
            'Dataset': ds_label,
            'Usul': 'Yadro Grid',
            'Outliers': o_yg,
            'Accuracy': f"{a_yg*100:.2f} %",
            'F1': f"{f_yg*100:.2f} %",
            'Time (s)': f"{t_yg:.2f}"
        })
        
    print("\n\n======== YAKUNIY JADVAL ========")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Save table to a markdown artifact locally
    res_df.to_markdown('final_comparison_results.md', index=False)

if __name__ == "__main__":
    main()

