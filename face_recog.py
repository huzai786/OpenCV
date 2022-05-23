import cv2 
import face_recognition
import numpy as np
import os

KNOWN_FACE_DIR = 'known'
known_faces = [] 
known_names = []

for file in os.listdir(KNOWN_FACE_DIR):
    name = f'{file}'.split('.')[0]
    image = face_recognition.load_image_file(f'{KNOWN_FACE_DIR}/{file}')
    encoding = face_recognition.face_encodings(image)
    known_faces.append(encoding)
    known_names.append(name)

cap = cv2.VideoCapture(0)

process_this_frame = True

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, known_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

