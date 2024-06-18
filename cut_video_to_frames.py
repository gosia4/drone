import cv2
import os

video_path = r"C:/Users/gosia/OneDrive - vus.hr/validate.mp4"
output_folder = r"C:/Users/gosia/OneDrive - vus.hr/validate"

# Utworzenie folderu, jeśli nie istnieje
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Utworzenie obiektu VideoCapture
cap = cv2.VideoCapture(video_path)

# Sprawdzenie, czy wideo zostało poprawnie otwarte
if not cap.isOpened():
    print("Nie udało się otworzyć pliku wideo")
    exit()

# Pobranie liczby klatek wideo, aby określić odpowiedni format numeracji
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_digits = len(str(total_frames))

# Licznik klatek
frame_count = 0

# Odczytywanie kolejnych klatek z nagrania i zapisywanie ich jako obrazy
while True:
    ret, frame = cap.read()  # Odczytanie kolejnej klatki
    if not ret:
        print("Nie ma więcej klatek lub odczyt klatki się nie powiódł")
        break  # Jeśli nie ma więcej klatek, zakończ pętlę

    # Zapisanie klatki jako obrazu w folderze wyjściowym z numeracją alfabetyczną
    frame_filename = os.path.join(output_folder, f"free_3_frame_{frame_count:0{num_digits}}.jpg")
    if cv2.imwrite(frame_filename, frame):
        print(f"Zapisano: {frame_filename}")
    else:
        print(f"Nie udało się zapisać: {frame_filename}")

    frame_count += 1

# Zwolnienie zasobów
cap.release()

# Wyświetlenie liczby zapisanych obrazów
print("Liczba zapisanych obrazów:", frame_count)












# # do dzielenia mp4
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#
# # Ścieżka do pliku wejściowego
# input_file = r"C:/Users/gosia/OneDrive - vus.hr/przyciete_35_obrócone.mp4"
#
# # Ścieżka do pliku wyjściowego
# output_file = r"C:/Users/gosia/OneDrive - vus.hr/validate.mp4"
#
# # Czas początkowy w sekundach (np. 10 sekund)
# start_time = 241
#
# # Czas końcowy w sekundach (np. 20 sekund)
# end_time = 480
#
# # Wycinanie fragmentu
# ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)


# 10-240 - train
# 241-480 validate