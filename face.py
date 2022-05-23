import os 
import cv2
import face_recognition

KNOWN_FACE_DIR = 'known'
TOLERANCE = 0.5
FRAME_THICKNESS = 1
FONT_THICKNESS = 1
MODEL = 'hog' #cnn
print('loading known images')

known_faces = [] 
known_names = []

for file in os.listdir(KNOWN_FACE_DIR):
    name = f'{file}'.split('.')[0]
    image = face_recognition.load_image_file(f'{KNOWN_FACE_DIR}/{file}')
    encoding = face_recognition.face_encodings(image)
    known_faces.append(encoding)
    known_names.append(name)

print(known_names)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    locations = face_recognition.face_locations(frame, model=MODEL)
    encodings = face_recognition.face_encodings(frame, locations)

    for encoding, location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, encoding, TOLERANCE)
        print(results)
        match = None
        if True in any(results):
            match = known_names[results.index(True)]
            print(f'match found: {match}')
            top_left = (location[3], location[0])
            bottom_right = (location[1], location[2])
            color = (0, 255, 0)
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (location[3], location[0])
            bottom_right = (location[1], location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(frame, match, (location[3]+10, location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200, 200), FONT_THICKNESS)

    cv2.imshow('cap', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
