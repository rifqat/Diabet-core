# Heart Disease Datasets - Column Information (Ustunlar haqida ma'lumot)

Ushbu papkadagi `processed_cleveland.csv` va `statlog_heart.dat` fayllarida jami **14 ta ustun (column)** mavjud. Qulaylik uchun ularning ma'nolari quyida keltirilgan:

| No | Ustun / Tashxis belgisi (Feature) | Ma'nosi (English) | Ma'nosi (O'zbekcha) |
|---|---|---|---|
| 1 | **age** | Age in years | Bemorning yoshi |
| 2 | **sex** | Sex (1 = male; 0 = female) | Jinsi (1 = erkak, 0 = ayol) |
| 3 | **cp** | Chest pain type (1, 2, 3, 4) | Ko'krak qafasi og'rig'i turi (4 tadan biri) |
| 4 | **trestbps** | Resting blood pressure (in mm Hg) | Tinch holatdagi qon bosimi |
| 5 | **chol** | Serum cholestoral in mg/dl | Xolesterin miqdori |
| 6 | **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | Qondagi qand miqdori (120 dan kattami?) |
| 7 | **restecg** | Resting electrocardiographic results (0, 1, 2) | EKG xulosasi (0, 1 yoki 2) |
| 8 | **thalach** | Maximum heart rate achieved | Maksimal yurak urish tezligi |
| 9 | **exang** | Exercise induced angina (1 = yes; 0 = no) | Jismoniy harakat vaqtidagi og'riq |
| 10 | **oldpeak** | ST depression induced by exercise relative to rest | Jismoniy zo'riqishdan keyingi EKG o'zgarishi |
| 11 | **slope** | The slope of the peak exercise ST segment (1, 2, 3) | ST segmenti qiyaligi (1, 2 yoki 3) |
| 12 | **ca** | Number of major vessels (0-3) colored by flourosopy | Katta qon tomirlari soni |
| 13 | **thal** | 3 = normal; 6 = fixed defect; 7 = reversable defect | Talassemiya (Yurak faoliyati nuqsonlari) |
| 14 | **target (class)** | Target class (Prediction) | **Natija:** Kasallik bormi yoki yo'q? |

---

### Python/Pandas da o'qish uchun kod nusxasi:

Agar ushbu fayllarni Python da tahlil qilmoqchi bo'lsangiz, `columns` argumentini berish orqali ustun nomlarini o'qishingiz mumkin.

**Cleveland CSV uchun (vergul bilan ajratilgan):**
```python
import pandas as pd

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df_cleveland = pd.read_csv('datasets/processed_cleveland.csv', names=columns, na_values='?')
```

**Statlog DAT uchun (bo'sh joy - probel bilan ajratilgan):**
```python
df_statlog = pd.read_csv('datasets/statlog_heart.dat', sep=' ', names=columns)
```

**Qayd:** 
Ba'zi joylarda (ayniqsa Cleveland faylida) baholash jarayonida belgilanmagan qiymatlar `?` xati bilan kiritilgan. Shu sababli uni o'qiyotganda `na_values='?'` parametridan foydalanish tavsiya etiladi.
