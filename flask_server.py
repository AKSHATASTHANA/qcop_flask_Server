from tkinter import Image

import cv2
import numpy as np
from face_recognition import recognize_objects_and_faces
from flask import Flask, request, jsonify



if __name__ == "__main__":
    main()
    app = Flask(__name__)


    @app.route('/')
    def index():
        return "Hello, World!"

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