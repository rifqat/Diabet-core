import urllib.request
import zipfile
import os

print("Datasetlarni yuklab olish boshlandi...")

# 1. Diabetes data (Dr John Schorling)
# Vanderbilt universitetining datasetlar omboridan olamiz (bu o'sha 403 qatorli dataset)
url1 = "https://hbiostat.org/data/repo/diabetes.csv"
try:
    urllib.request.urlretrieve(url1, "dataset1_diabetes.csv")
    print("Dataset 1 (Diabetes) muvaffaqiyatli yuklab olindi.")
except Exception as e:
    print(f"Dataset 1 yuklashda xatolik: {e}")

# 2. Men's dataset (Golino et al, Figshare)
# Figshare API orqali yoki to'g'ridan to'g'ri download linki orniga qidiramiz.
url2 = "https://figshare.com/ndownloader/files/1381368" # Men's dataset CSV
try:
    req = urllib.request.Request("https://figshare.com/ndownloader/files/1381368", headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open("dataset2_hypertension.csv", 'wb') as out_file:
        out_file.write(response.read())
    print("Dataset 2 (Hypertension) muvaffaqiyatli yuklab olindi.")
except Exception as e:
    print(f"Dataset 2 yuklashda xatolik: {e}")

# 3. Chronic Kidney Disease (UCI)
url3 = "https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"
try:
    urllib.request.urlretrieve(url3, "dataset3_ckd.zip")
    with zipfile.ZipFile("dataset3_ckd.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset3_ckd")
    print("Dataset 3 (Chronic Kidney Disease) muvaffaqiyatli yuklab olindi va arxivdan chiqarildi.")
    os.remove("dataset3_ckd.zip")
except Exception as e:
    print(f"Dataset 3 yuklashda xatolik: {e}")

print("So'ralgan barcha ma'lumotlar papkaga yuklandi!")
