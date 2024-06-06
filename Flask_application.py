from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_scss import Scss
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
Bootstrap(app)
Scss(app, static_dir='static/css', asset_dir='static/scss')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
