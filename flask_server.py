from flask import Flask, request, jsonify
from PIL import Image
import cv2
import numpy as np
from face_recognition import recognize_objects_and_faces
from flasgger import Swagger, swag_from

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def index():
    """
    Index endpoint
    ---
    responses:
      200:
        description: Returns a greeting message
    """
    return "Hello, World!"

@app.route('/recognize', methods=['POST'])
@swag_from({
    'responses': {
        200: {
            'description': 'Upload an image for object and face recognition',
            'schema': {
                'type': 'object',
                'properties': {
                    'objects': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'name': {'type': 'string'},
                                'confidence': {'type': 'number'}
                            }
                        }
                    },
                    'faces': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'name': {'type': 'string'},
                                'confidence': {'type': 'number'}
                            }
                        }
                    }
                }
            }
        },
        400: {
            'description': 'No image provided',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'The image file to be uploaded'
        }
    ]
})
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    collection_id = "new_face_collection"
    objects, faces = recognize_objects_and_faces(image, collection_id)

    return jsonify({"objects": objects, "faces": faces})

@app.route('/api', methods=['GET'])
def api_page():
    """
    API Information
    ---
    responses:
      200:
        description: Returns information about the API and its endpoints
    """
    return jsonify({
        "message": "Welcome to the Object and Face Recognition API",
        "endpoints": {
            "POST /recognize": "Upload an image for object and face recognition"
        }
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)