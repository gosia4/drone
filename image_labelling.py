import time
import cv2
import numpy as np
import os
import glob

# to działa, ale zapisuje w czterech linijkach
def tag_obrazy(folder_path, output_folder):
    """
    Function to tag images in a folder and save coordinates to files.

    Arguments:
        folder_path: Path to the folder containing images.
        output_folder: Path to the folder where files with coordinates will be saved.

    Returns:
        List containing coordinates of all drawn rectangles for all images.
    """

    # List to store coordinates
    obszary_zainteresowania = []

    # Get list of files in the folder
    pliki = os.listdir(folder_path)

    # Iterate over files
    for plik in pliki:
        # Check if the file is an image
        if plik.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            img = cv2.imread(os.path.join(folder_path, plik))

            # Call function to mark areas
            obszary = tag_klatki(img, plik)

            # Add coordinates to the list
            obszary_zainteresowania.extend(obszary)

            # Save coordinates to a file
            nazwa_pliku = os.path.splitext(plik)[0] + '.txt'
            with open(os.path.join(output_folder, nazwa_pliku), 'w') as f:
                for obszar in obszary:
                    # Convert coordinates to a NumPy array before writing
                    obszar_array = np.array(obszar)
                    # Write array as a single line separated by spaces
                    f.write(' '.join(map(str, obszar_array)) + ' ')

    return obszary_zainteresowania


def tag_klatki(img, nazwa_pliku):
    """
    Function to mark areas on an image.

    Arguments:
        img: Image to be tagged.
        nazwa_pliku: Nazwa pliku, która będzie wyświetlana w oknie.

    Returns:
        List containing coordinates of all drawn rectangles.
    """
    img_copy = None
    obszary = []

    def zaznacz_obszar(event, x, y, flags, param):
        nonlocal img_copy, obszary
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append first point (x, y)
            obszary.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(obszary) > 0:
                # Append second point (x, y)
                obszary.append([x, y])

    cv2.namedWindow(nazwa_pliku)  # Ustawiamy nazwę okna jako nazwa pliku
    cv2.setMouseCallback(nazwa_pliku, zaznacz_obszar)

    while True:
        img_copy = img.copy()
        cv2.imshow(nazwa_pliku, img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return obszary



def zapisz_do_pliku(obszary, output_folder):
    """
    Funkcja zapisuje współrzędne do plików w formacie zgodnym z repozytorium.

    Argumenty:
        obszary: Lista zawierająca współrzędne obszarów.
        output_folder: Ścieżka do folderu, w którym zostaną zapisane pliki.

    Returns:
        None.
    """
    # os.path.splitext(plik)[0] + '.txt'
    nazwa_pliku = f"{int(time.time())}.txt"
    with open(os.path.join(output_folder, nazwa_pliku), 'w') as f:
        for obszar in obszary:
            # Zapisz wszystkie współrzędne w jednej linii
            f.write(' '.join(map(str, obszar)))
    # for obszar in obszary:
    #     # Utworzenie nazwy pliku
    #     nazwa_pliku = f"{int(time.time())}.txt"
    #     # Zapis współrzędnych do pliku
    #     with open(os.path.join(output_folder, nazwa_pliku), 'w') as f:
    #         f.write(' '.join(map(str, obszar)))


# normalizacja współrzędnych:
def konwertuj_wspolrzedne(plik_txt, output_folder, image_width, image_height):
    with open(plik_txt, 'r') as f:
        wspolrzedne = f.readlines()

    yolo_wspolrzedne = []

    for wsp in wspolrzedne:
        if len(wsp.split()) < 4:
            print(f"Plik '{plik_txt}' zawiera mniej niż 4 liczby i zostanie pominięty.")
            continue

        x1, y1, x2, y2 = map(float, wsp.split())

        # Obliczanie środka prostokąta
        srodek_x = (x1 + x2) / 2
        srodek_y = (y1 + y2) / 2

        # Obliczanie szerokości i wysokości prostokąta
        szerokosc = abs(x2 - x1)
        wysokosc = abs(y2 - y1)

        # Normalizacja współrzędnych do zakresu [0, 1] względem szerokości i wysokości obrazu
        x1_norm = srodek_x / image_width
        y1_norm = srodek_y / image_height
        szerokosc_norm = szerokosc / image_width
        wysokosc_norm = wysokosc / image_height

        klasa = 0
        # Format YOLO: klasa środek_x środek_y szerokość wysokość
        yolo_wspolrzedne.append([int(klasa), x1_norm, y1_norm, szerokosc_norm, wysokosc_norm])


    nazwa_pliku = os.path.splitext(os.path.basename(plik_txt))[0] + '.txt'
    with open(os.path.join(output_folder, nazwa_pliku), 'w') as f:
        for wsp in yolo_wspolrzedne:
            f.write(' '.join(map(str, wsp)) + '\n')


# Ustawienia do konwertowania wspórzédnych

# folder_txt = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\free_3_niezmormalizowane, validate"
# folder_txt = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\train\labels\free_1_przed_konwertowaniem"
# # output_folder = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone-detection\drone-detection\thermographic_data\validate\labels\free_3"
# output_folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\train\labels\free_1"
#
# # Konwersja plików txt
# for plik in os.listdir(folder_txt):
#     if plik.endswith('.txt'):
#         konwertuj_wspolrzedne(os.path.join(folder_txt, plik), output_folder, 640, 480)




# Ustawienie ścieżki do folderu z obrazami, tagowanie obrazów

# folder_path = r"C:\Users\gosia\OneDrive - vus.hr\klatki_przyciete_7mb\test"
# folder_path = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone-detection-thermal-images\thermal_signature_drone_detection\thermographic_day\validate\images"
folder_path = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\images\free_3"

# Ustawienie ścieżki do folderu do zapisu współrzędnych
output_folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\labels\nieprzekonwertowane"
# output_folder = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\probne_tagowanie"
# output_folder = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\drone-detection-thermal-images\thermal_signature_drone_detection\thermographic_day\train\labels_2"

# Wywołanie funkcji do tagowania obrazów
obszary_zainteresowania = tag_obrazy(folder_path, output_folder)

# Wyświetlenie współrzędnych zaznaczonych obszarów
# print("Współrzędne zaznaczonych obszarów:")
# for obszar in obszary_zainteresowania:
#     print(obszar)





# zamień mając już współrzędne lewego górnego rogu oraz prawego dolnego na format yolo,
# czyli współrzędne środka oraz wysokość i szerkość bounding boxa
import os
import glob

def konwertuj_wspolrzedne2(folder_path, output_folder, image_width, image_height):
    """
    Funkcja konwertująca współrzędne prostokątów na format YOLO.

    Arguments:
        folder_path: Ścieżka do folderu zawierającego pliki tekstowe z współrzędnymi.
        output_folder: Ścieżka do folderu, w którym zostaną zapisane pliki w formacie YOLO.
        image_width: Szerokość obrazu.
        image_height: Wysokość obrazu.

    Returns:
        None.
    """
    # Utwórz folder docelowy, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Pobierz listę plików zgodnych z danym wzorcem nazwy
    files = glob.glob(os.path.join(folder_path, "*.txt"))

    # Iteruj przez wszystkie pliki
    for file_path in files:
        with open(file_path, 'r') as f:
            wspolrzedne = f.readlines()

        yolo_wspolrzedne = []

        for wsp in wspolrzedne:
            if len(wsp.split()) < 4:
                print(f"Plik '{file_path}' zawiera mniej niż 4 liczby i zostanie pominięty.")
                continue
            x1, y1, x2, y2 = map(float, wsp.split())

            # Obliczanie współrzędnych środka prostokąta
            srodek_x = (x1 + x2) / 2
            srodek_y = (y1 + y2) / 2

            # Obliczanie szerokości i wysokości prostokąta
            szerokosc = abs(x2 - x1)
            wysokosc = abs(y2 - y1)

            # Normalizacja współrzędnych do zakresu [0, 1] względem szerokości i wysokości obrazu
            x1_norm = round(srodek_x / image_width, 6)
            y1_norm = round(srodek_y / image_height, 6)
            szerokosc_norm = round(szerokosc / image_width, 6)
            wysokosc_norm = round(wysokosc / image_height, 6)

            klasa = 0
            # Format YOLO: klasa środek_x środek_y szerokość wysokość
            yolo_wspolrzedne.append([int(klasa), x1_norm, y1_norm, szerokosc_norm, wysokosc_norm])

        # Utwórz nazwę pliku YOLO
        nazwa_pliku = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
        # Zapisz współrzędne do pliku YOLO
        with open(os.path.join(output_folder, nazwa_pliku), 'w') as f:
            for wsp in yolo_wspolrzedne:
                f.write(' '.join(map(str, wsp)) + '\n')


# folder = r"C:\Users\gosia\drone-detection\drone-detection\do_testu\free_1_przed_konwertowaniem"
folder = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\labels\nieprzekonwertowane"
# output = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\train\labels\free_1"
output = r"C:\Users\gosia\drone-detection\drone-detection\thermographic_data\validate\labels\free_3"

konwertuj_wspolrzedne2(folder, output, 416, 416)
