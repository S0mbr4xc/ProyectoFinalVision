import os

NEG_DIR = "dataset-lbp/negatives"
BG_FILE = "dataset-lbp/bg.txt"

with open(BG_FILE, "w") as f:
    for root, _, files in os.walk(NEG_DIR):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                abs_path = os.path.abspath(os.path.join(root, name))
                f.write(f"{abs_path}\n")

print(f"✅ bg.txt generado con {sum(1 for _ in open(BG_FILE))} imágenes negativas.")


