import face_recognition

obama_pic = face_recognition.load_image_file('assets/obama.jpeg')
obama_grp_pic = face_recognition.load_image_file('assets/obama_grp.jpg')
obama_pic2 = face_recognition.load_image_file('assets/obama3.jpg')
not_obama = face_recognition.load_image_file('assets/trump.jpg')
group = face_recognition.load_image_file('assets/grp.jpg')

obama_face_locations = face_recognition.face_encodings(obama_pic)[0]
obama2_face_locations = face_recognition.face_encodings(obama_pic2)[0]
unknown_face_locations = face_recognition.face_encodings(not_obama)[0]
group_face_locations = face_recognition.face_encodings(group)
obama_grp = face_recognition.face_encodings(obama_grp_pic)

print(obama_grp)
results = face_recognition.compare_faces(obama_grp, obama_face_locations)
print(results)
if any(results) == True:
    print('yes we got him')
else:
    print('obama not found')
# print(obama_face_locations)
# print(obama2_face_locations)
# results = face_recognition.compare_faces([obama_face_locations], obama2_face_locations)
# print(results)