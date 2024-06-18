import os
import cv2

def resize_image(image, target_size):
    """Funkcja do przeskalowania pojedynczego obrazu."""
    return cv2.resize(image, target_size)

def resize_images_in_folder(input_folder, output_folder, target_size):
    """Funkcja do przeskalowania obrazów w danym folderze do określonego rozmiaru."""
    # Upewnij się, że folder docelowy istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Przeskaluj każdy obraz w folderze wejściowym
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Wczytaj obraz
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Przeskaluj obraz
            resized_image = resize_image(image, target_size)

            # Zapisz przeskalowany obraz do folderu docelowego
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)

            print(f"Przeskalowano i zapisano: {output_path}")

# Przykładowe użycie resizing image
input_folder = r"C:\Users\gosia\OneDrive - vus.hr\train"
output_folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\train\images\free_1"
target_size = (416, 416)

resize_images_in_folder(input_folder, output_folder, target_size)


def resize_bounding_box(bbox, original_size, target_size):
    """Funkcja do przeskalowania bounding boxa."""
    # Pobierz oryginalne wymiary obrazu
    original_height, original_width = original_size

    # Pobierz docelowe wymiary obrazu
    target_height, target_width = target_size

    # Wylicz współczynniki przeskalowania dla wysokości i szerokości
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Przeskaluj współrzędne bounding boxa
    # bbox_resized = [
    #     bbox[1] * scale_x,  # Nowa współrzędna x środka bounding boxa
    #     bbox[2] * scale_y,  # Nowa współrzędna y środka bounding boxa
    #     bbox[3] * scale_x,  # Nowa szerokość bounding boxa
    #     bbox[4] * scale_y  # Nowa wysokość bounding boxa
    # ]
    bbox_resized = [
        round(bbox[0] * scale_x, 6),  # Nowa współrzędna x środka bounding boxa
        round(bbox[1] * scale_y, 6),  # Nowa współrzędna y środka bounding boxa
        round(bbox[2] * scale_x, 6),  # Nowa szerokość bounding boxa
        round(bbox[3] * scale_y, 6)  # Nowa wysokość bounding boxa
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


# Ścieżka do folderu z otagowanymi obrazami
# folder_path = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\labels\free_3"

# # Przetwarzaj każdy plik w folderze
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         # Pełna ścieżka do pliku tekstowego z otagowanymi bounding boxami
#         txt_file = os.path.join(folder_path, filename)
#
#         # Przeskaluj bounding boxy
#         bboxes = load_bounding_boxes_from_txt(txt_file)
#         original_image_size = (640, 480)
#         target_image_size = (416, 416)
#         resized_bboxes = [resize_bounding_box(bbox, original_image_size, target_image_size) for bbox in bboxes]
#
#         # Zapisz przeskalowane bounding boxy do pliku tekstowego
#         save_bounding_boxes_to_txt(resized_bboxes, txt_file)





# Przetwarzaj każdy plik w folderze
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         # Pełna ścieżka do pliku tekstowego
#         txt_file = os.path.join(folder_path, filename)
#
#         # Licznik łańcuchów w każdej linii
#         line_string_count = []
#
#         # Otwórz plik i przetwarzaj linie
#         with open(txt_file, 'r') as file:
#             for line in file:
#                 # Podziel linię na łańcuchy (przestrzeń jest separatorem)
#                 strings_in_line = line.strip().split()
#
#                 # Dodaj liczbę łańcuchów w linii do listy
#                 line_string_count.append(len(strings_in_line))
#
#         # Wyświetl informacje o liczbie łańcuchów w każdej linii w pliku
#         print(f"Liczba łańcuchów w pliku {filename}: {line_string_count}")
