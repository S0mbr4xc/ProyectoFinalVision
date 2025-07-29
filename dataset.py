import os
import zipfile
import shutil
import tarfile
from torchvision.datasets import Places365
from tqdm import tqdm

# 📁 Directorios base
base_dir = "dataset-lbp"
pos_dir = os.path.join(base_dir, "positives")
neg_dir = os.path.join(base_dir, "negatives")
ann_dir = os.path.join(base_dir, "annotations")
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

# ─── PARTE 1: WIDER FACE ───
wider_zip = "WIDER_train.zip"
if os.path.exists(wider_zip):
    print("📦 Extrayendo WIDER FACE...")
    with zipfile.ZipFile(wider_zip, 'r') as z:
        z.extractall(os.path.join(base_dir, "wider_raw"))
    # Copiar imágenes
    for root, _, files in os.walk(os.path.join(base_dir, "wider_raw", "WIDER_train", "images")):
        for f in files:
            if f.lower().endswith(".jpg"):
                shutil.copy2(os.path.join(root, f), os.path.join(pos_dir, f))
    # Copiar anotaciones
    ann_src = os.path.join(base_dir, "wider_raw", "wider_face_split", "wider_face_train_bbx_gt.txt")
    if os.path.exists(ann_src):
        shutil.copy2(ann_src, os.path.join(ann_dir, "wider_face_train_bbx_gt.txt"))
    print("✅ WIDER FACE listo.")
else:
    print("⚠️ No encontré el archivo WIDER_train.zip en este directorio.")

# ─── PARTE 2: Places365 negativos ───
print("⬇️ Descargando Places365 (versión small)...")
dataset = Places365(root=base_dir, split="train-standard", small=True, download=True)
print("📥 Places365 descargado, moviendo imágenes...")

for img, label in tqdm(dataset.imgs, desc="Copiando negativos"):
    src = img
    dst = os.path.join(neg_dir, os.path.basename(src))
    shutil.copy2(src, dst)

print("✅ Imágenes negativas Preparadas.")

print(f"🎯 Dataset organizado en:\n- Positivos: {len(os.listdir(pos_dir))} imágenes\n- Negativos: {len(os.listdir(neg_dir))} imágenes")
