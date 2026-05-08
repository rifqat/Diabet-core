import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class HybridPredictionModel:
    """
    Maqolada ko'rsatilgan DBSCAN + SMOTE + Random Forest usullarini
    birlashtirgan Gibrid bashorat Qilish Modeli (HPM).
    """
    def __init__(self, eps=0.5, min_samples=5, smote_ratio='auto', rf_estimators=100, cv_folds=10):
        # DBSCAN parametrlari
        self.eps = eps
        self.min_samples = min_samples
        # SMOTE parametrlari
        self.smote_ratio = smote_ratio
        # Random Forest parametrlari
        self.rf_estimators = rf_estimators
        # Cross validation uchun qismlar soni (Maqolada 10-fold ko'rsatilgan)
        self.cv_folds = cv_folds
        
    def train_and_evaluate(self, X, y):
        """
        X: pandas DataFrame yoki numpy array (o'zgaruvchilar masalan yosh, qon bosimi)
        y: pandas Series yoki numpy array (natijalar masalan 0 sog'lom, 1 kasal)
        """
        print(f"[1] Dastlabki ma'lumotlar hajmi: {X.shape}")
        
        # 1-QADAM: DBSCAN bilan xato ma'lumotlarni (outliers) tozalash
        # DBSCAN to'g'ri ishlashi uchun ma'lumotlarni standartlashtiramiz (scale)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # DBSCAN'da -1 qiymat xato (outlier) ekanligini bildiradi. 
        # Shuning uchun klasterga tushmagan (outlier) qatorlarni olib tashlaymiz.
        mask = clusters != -1
        X_clean = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        y_clean = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
        
        print(f"[2] DBSCAN tozalagandan keyingi hajm: {X_clean.shape}")
        
        # 2-QADAM: SMOTE bilan ma'lumotlarni balansga keltirish (oversampling)
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
        
        print(f"[3] SMOTE tenglashtirgandan keyingi hajm: {X_resampled.shape}")
        
        # 3-QADAM: Random Forest orqali klassifikatsiya va baholash
        rf = RandomForestClassifier(n_estimators=self.rf_estimators, random_state=42)
        
        # Maqolada aytilganidek 10-Fold Cross Validation qo'llaymiz
        print(f"[4] Random Forest modeli {self.cv_folds}-Fold bo'yicha o'rganmoqda va baholamoqda...")
        y_pred = cross_val_predict(rf, X_resampled, y_resampled, cv=self.cv_folds)
        
        # Modellarni baholash metrikalarini hisoblaymiz
        tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()
        
        accuracy = accuracy_score(y_resampled, y_pred)
        precision = precision_score(y_resampled, y_pred)
        recall = recall_score(y_resampled, y_pred) # Boshqacha nomi: Sensitivity
        f1 = f1_score(y_resampled, y_pred)
        specificity = tn / (tn + fp)
        
        print("\n=== Modelni Baholash Natijalari (Maqoladagidek) ===")
        print(f"Aniqligi (Accuracy):    {accuracy * 100:.3f} %")
        print(f"Ta'sirchanlik (Recall): {recall * 100:.3f} %")
        print(f"Maxsusligi (Specificity): {specificity * 100:.3f} %")
        print(f"F1 Qiymati (F1 Score):  {f1 * 100:.3f} %")
        print(f"Keng qamrov (Precision):{precision * 100:.3f} %")
        
        # O'rganilgan modelni qaytaramiz (kelajakdagi yangi bemorlar uchun ishlatishga)
        rf.fit(X_resampled, y_resampled)
        return rf

# Pastdagi qism qanday ishlatishni ko'rsatish uchun NAMUNA:
if __name__ == "__main__":
    print("Gibrid Model (HPM) Skripti Ishga tushirildi.\n")
    # Misol uchun test ma'lumotlarini yaratamiz yoki yuklaymiz:
    # haqiqiy holatda df = pd.read_csv("kasallar_bazas.csv") qilib olinadi.
    
    # Namuna (Fake Data): 500 ta sog'lom, 50 ta kasal odam ma'lumoti 
    # (balans buzilgan ko'rinishda)
    from sklearn.datasets import make_classification
    
    X_fake, y_fake = make_classification(n_samples=550, n_features=5, 
                                         weights=[0.9, 0.1], random_state=42)
    
    X_fake_df = pd.DataFrame(X_fake, columns=['yosh', 'qon_bosimi', 'vazn', 'boy', 'shakar'])
    y_fake_s = pd.Series(y_fake, name='Kasal_yoki_soglom')
    
    # Modelni chaqirib sozlaymiz
    hpm_model = HybridPredictionModel(eps=1.5, min_samples=5)
    
    # Modelni ishga tushirish:
    # hpm_model.train_and_evaluate(X_fake_df, y_fake_s)
    print("Namunaviy ma'lumotlarni test qilib ko'rish uchun yuqoridagi 'hpm_model.train_and_evaluate(X_fake_df, y_fake_s)' qatoridan komentariyani olib tashlang.")
