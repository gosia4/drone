import os
import cv2

def resize_image(image, target_size):
    """Funkcja do przeskalowania pojedynczego obrazu."""
    return cv2.resize(image, target_size)

def resize_images_in_folder(input_folder, output_folder, target_size):
    """Funkcja do przeskalowania obrazów w danym folderze do określonego rozmiaru."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            resized_image = resize_image(image, target_size)

            # Zapisz przeskalowany obraz do folderu docelowego
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)

            print(f"Przeskalowano i zapisano: {output_path}")

input_folder = r"ścieżka_do_folderu_z_obrazami_do_zmiany_rozmiaru"
output_folder = r"ścieżka_do_folderu_docelowego"
target_size = (416, 416)

resize_images_in_folder(input_folder, output_folder, target_size)


def resize_bounding_box(bbox, original_size, target_size):
    """Funkcja do przeskalowania bounding boxa."""

    original_height, original_width = original_size

    target_height, target_width = target_size

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    bbox_resized = [
        round(bbox[0] * scale_x, 6),  # współrzędna x środka bounding boxa
        round(bbox[1] * scale_y, 6),  # współrzędna y środka bounding boxa
        round(bbox[2] * scale_x, 6),  # szerokość bounding boxa
        round(bbox[3] * scale_y, 6)  # wysokość bounding boxa
    ]

    return bbox_resized


def load_bounding_boxes_from_txt(txt_file):
    """Funkcja do wczytania otagowanych bounding boxów z pliku tekstowego."""
    bboxes = []
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # parts[0] to klasa, parts[1:] to współrzędne x, y, wysokość i szerokość bounding boxa
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
    return bboxes


def save_bounding_boxes_to_txt(bboxes, txt_file):
    """Funkcja do zapisania przeskalowanych bounding boxów do pliku tekstowego."""
    with open(txt_file, 'w') as file:
        for bbox in bboxes:
            line = ' '.join(map(str, bbox)) + '\n'
            file.write(line)

# Skalowanie otagowanych obrazów
# Ścieżka do folderu z otagowanymi obrazami
# folder_path = r"ścieżka_do_folderu_z_otagowanymi_obrazami"

# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         # Pełna ścieżka do pliku tekstowego z otagowanymi bounding boxami
#         txt_file = os.path.join(folder_path, filename)
#         bboxes = load_bounding_boxes_from_txt(txt_file)
#         original_image_size = (640, 480)
#         target_image_size = (416, 416)
#         resized_bboxes = [resize_bounding_box(bbox, original_image_size, target_image_size) for bbox in bboxes]
#         save_bounding_boxes_to_txt(resized_bboxes, txt_file)



