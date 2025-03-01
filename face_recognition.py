import boto3
import cv2
import io
from PIL import Image, ImageDraw
from botocore.exceptions import ClientError
import numpy as np
from flask import Flask, request, jsonify

image_to_name = {
    "image1.jpg": "Akshat",
    "image2.jpg": "Akshat"
}

def create_collection_if_not_exists(collection_id):
    client = boto3.client('rekognition')
    try:
        response = client.create_collection(CollectionId=collection_id)
        print(f'Collection {collection_id} created: {response}')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            print(f'Collection {collection_id} already exists.')
        else:
            raise

def add_faces_to_collection(bucket, photo, collection_id):
    client = boto3.client('rekognition')
    response = client.index_faces(
        CollectionId=collection_id,
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        ExternalImageId=photo,
        DetectionAttributes=['ALL']
    )
    print(f'Faces added to collection {collection_id}: {response}')

def recognize_objects_and_faces(image, collection_id):
    client = boto3.client('rekognition')
    img_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    response_objects = client.detect_labels(
        Image={'Bytes': img_bytes},
        MaxLabels=10,
        MinConfidence=75
    )

    objects = []
    if 'Labels' in response_objects:
        for label in response_objects['Labels']:
            name = label['Name']
            confidence = label['Confidence']
            print(f'Detected object: {name} with confidence: {confidence}')
            objects.append((name, confidence))

    response_faces = client.search_faces_by_image(
        CollectionId=collection_id,
        Image={'Bytes': img_bytes},
        FaceMatchThreshold=95,
        MaxFaces=5
    )

    faces = []
    if 'FaceMatches' in response_faces:
        for match in response_faces['FaceMatches']:
            face = match['Face']
            face_id = face['ExternalImageId']
            name = image_to_name.get(face_id, "Person")
            print(f'Matched face: {name} with confidence: {match["Similarity"]}')
            faces.append((name, match['Similarity']))

    return objects, faces

def main():
    bucket = "qcopbucket"
    photos = ["image1.jpg", "image2.jpg"]
    collection_id = "new_face_collection"

    create_collection_if_not_exists(collection_id)

    for photo in photos:
        add_faces_to_collection(bucket, photo, collection_id)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_region = frame[y:y+h, x:x+w]
            objects, recognized_faces = recognize_objects_and_faces(face_region, collection_id)

            for obj_name, confidence in objects:
                if obj_name.lower() in ['bottle', 'hat']:
                    cv2.putText(frame, f'{obj_name} ({confidence:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            for face_id, similarity in recognized_faces:
                cv2.putText(frame, f'{face_id} ({similarity:.2f}%)', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Real-time Object and Face Recognition - Bottle, Hat, and Person Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    app = Flask(__name__)

    @app.route('/recognize', methods=['POST'])
    def recognize():
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image = Image.open(image_file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        collection_id = "new_face_collection"
        objects, faces = recognize_objects_and_faces(image, collection_id)

        return jsonify({"objects": objects, "faces": faces})

    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000)