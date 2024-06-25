import cv2
import os

input_folder = r"ścieżka_do folderu_z_obrazami"
output_folder = r"Ścieżka_do_folderu_wyjśćiowego"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)

        image = cv2.imread(input_path)

        if image is not None:
            # Przekształć obraz 8-bit
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, gray_image)
        else:
            print(f"Błąd podczas wczytywania obrazu: {input_path}")
