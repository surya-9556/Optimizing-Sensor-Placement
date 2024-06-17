from flask import Flask, render_template, request, jsonify, make_response
from main import OptimizingPlacement
from flask_bootstrap import Bootstrap
from flask_scss import Scss
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pandas as pd
import os

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
Bootstrap(app)
Scss(app, static_dir='static/css', asset_dir='static/scss')

placement = OptimizingPlacement()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def reading_file():
    if 'file' not in request.files:
        response = {'Message': 'No file part in the request', 'Status': 400}
        return render_template('upload.html', result=response)

    file = request.files['file']

    if file.filename == '':
        response = {'Message': 'No selected file', 'Status': 400}
        return render_template('upload.html', result=response)
    
    # Secure the filename and save the file locally
    filename = secure_filename(file.filename)
    save_path = os.path.join('Uploads', filename)
    os.makedirs('Uploads', exist_ok=True)
    file.save(save_path)

    print(f'File saved to: {save_path}')

    try:
        # Perform operations with OptimizingPlacement class
        result_head, keylist = placement.extracting_files(save_path, 'Uploads/Sensor Data')

        if isinstance(result_head, pd.DataFrame):
            response = {
                'Message': 'File successfully uploaded',
                'File_size': os.path.getsize(save_path),
                'Head_of_DataFrame': result_head.head().to_dict(),
                'Status': 200
            }

        else:
            response = {'Message': 'Error processing file', 'Status': 500}
    except Exception as e:
        response = {'Message': str(e), 'Status': 500}
    
    return render_template('upload.html', result=response)

if __name__ == '__main__':
    app.run(debug=True)
