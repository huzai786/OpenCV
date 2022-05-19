import cv2 
import face_recognition
import numpy as np


cap = cv2.VideoCapture('assets/trumpwput.mp4')


obama_face = face_recognition.load_image_file('assets/obama.jpeg')
putin_face = face_recognition.load_image_file('assets/vlad.jpeg')
trump_face = face_recognition.load_image_file('assets/trump3.jpeg')
obama_face_encoding = face_recognition.face_encodings(obama_face)[0]
trump_face_encoding = face_recognition.face_encodings(trump_face)[0]
putin_face_encoding = face_recognition.face_encodings(putin_face)[0]



known_face_encodings = [
    obama_face_encoding,
    trump_face_encoding,
    putin_face_encoding
]
known_face_names = [
    "obama",
    "Trump",
    'putin'
]

face_locations = []
face_encodings = []
face_names = []
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

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(90) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

