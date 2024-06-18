import cv2
import os

video_path = r"C:/Users/gosia/OneDrive - vus.hr/validate.mp4"
output_folder = r"C:/Users/gosia/OneDrive - vus.hr/validate"

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



# # do dzielenia mp4
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#
# input_file = r"C:/Users/gosia/OneDrive - vus.hr/przyciete_35_obrócone.mp4"
# output_file = r"C:/Users/gosia/OneDrive - vus.hr/validate.mp4"
#
# # Czas początkowy w sekundach (np. 10 sekund)
# start_time = 241
#
# # Czas końcowy w sekundach (np. 20 sekund)
# end_time = 480
# ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
