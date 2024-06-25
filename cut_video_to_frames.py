import cv2
import os

video_path = r"ścieżka_do_nagrania_wideo.mp4"
output_folder = r"Ścieżka_do_flderu_z_klatkami"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Nie udało się otworzyć pliku wideo")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_digits = len(str(total_frames))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie ma więcej klatek lub odczyt klatki się nie powiódł")
        break

    frame_filename = os.path.join(output_folder, f"free_3_frame_{frame_count:0{num_digits}}.jpg")
    if cv2.imwrite(frame_filename, frame):
        print(f"Zapisano: {frame_filename}")
    else:
        print(f"Nie udało się zapisać: {frame_filename}")

    frame_count += 1

cap.release()

print("Liczba zapisanych obrazów:", frame_count)

