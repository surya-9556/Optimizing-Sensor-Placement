from flask import Flask, render_template, request, jsonify, make_response
from main import OptimizingPlacement
from flask_bootstrap import Bootstrap
from flask_scss import Scss
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import os
import json

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
Bootstrap(app)
Scss(app, static_dir='static/css', asset_dir='static/scss')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def reading_file():
    if 'file' not in request.files:
        response = make_response(jsonify({'Message': 'No file part in the request'}), 400)
        response_data = response.get_json()
        response_data['Status'] = response.status_code
        response.set_data(json.dumps(response_data))
        return response
    
    file = request.files['file']

    if file.filename == '':
        response = make_response(jsonify({'Message': 'No selected file'}), 400)
        response_data = response.get_json()
        response_data['Status'] = response.status_code
        response.set_data(json.dumps(response_data))
        return response
    
    # Save the file to a temporary location
    temp_file = NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    file_path = temp_file.name

    print(f'path name: {file_path}')

    # Perform operations with OptimizingPlacement class
    placement = OptimizingPlacement()
    result_head = placement.extracting_files(file_path)
    
    # Delete the temporary file
    os.remove(file_path)

    if result_head is not None:
        response = make_response(jsonify({
            'Message': 'File successfully uploaded',
            'File_size': os.path.getsize(file_path),
            'Head_of_DataFrame': result_head.to_dict()
        }), 200)
    else:
        response = make_response(jsonify({
            'Message': 'Error processing file'
        }), 500)
    
    response_data = response.get_json()
    response_data['Status'] = response.status_code
    response.set_data(json.dumps(response_data))
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
