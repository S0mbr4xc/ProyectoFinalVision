import os
import cv2

# 📁 Rutas
ROOT_DIR = "C:/Users/s3_xc/OneDrive - Universidad Politecnica Salesiana/Uni/Programacion/Vision del computador/ProyectoFinal"
WIDER_IMG_DIR = os.path.join(ROOT_DIR, "dataset-lbp", "wider_raw", "WIDER_train", "images")
ANNOTATIONS_TXT = os.path.join(ROOT_DIR, "dataset-lbp", "annotations", "wider_face_train_bbx_gt.txt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset-lbp", "cropped_faces")
INFO_TXT = os.path.join(ROOT_DIR, "dataset-lbp", "info.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANNOTATIONS_TXT, "r") as file:
    lines = file.readlines()

with open(INFO_TXT, "w") as info_out:
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "":
            i += 1
            continue

        image_rel_path = line
        i += 1
        if i >= len(lines):
            break

        try:
            num_faces = int(lines[i].strip())
        except ValueError:
            print(f"❌ Error leyendo número de caras para {image_rel_path}. Saltando bloque.")
            i += 1
            continue

        i += 1
        image_path = os.path.join(WIDER_IMG_DIR, image_rel_path)
        if not os.path.exists(image_path):
            i += num_faces
            continue

        image = cv2.imread(image_path)
        if image is None:
            i += num_faces
            continue

        for face_idx in range(num_faces):
            if i >= len(lines):
                break
            bbox = lines[i].strip().split()
            i += 1
            if len(bbox) < 4:
                continue
            x, y, w, h = map(int, bbox[:4])
            if w < 20 or h < 20:
                continue

            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (100, 100))
            output_name = f"{os.path.splitext(os.path.basename(image_rel_path))[0]}_{face_idx}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(output_path, resized)
            info_out.write(f"{output_path} 1 0 0 100 100\n")

print("✅ Rostros recortados y archivo info.txt generado correctamente.")
