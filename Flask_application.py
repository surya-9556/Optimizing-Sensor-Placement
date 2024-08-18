from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_bootstrap import Bootstrap
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
Bootstrap(app)

# Set the folder for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    sample_data = None
    filename = request.args.get('filename')
    
    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            sample_data = df.head().to_html(classes='table table-striped', index=False)

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            sample_data = df.head().to_html(classes='table table-striped', index=False)
            
            return render_template('index.html', sample_data=sample_data, filename=filename)

    return render_template('index.html', sample_data=sample_data, filename=filename)

@app.route('/calculate_stats', methods=['GET', 'POST'])
def calculate_stats():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    columns = df.columns.tolist()
    stats = None
    hist_url = None
    box_url = None
    density_url = None

    if request.method == 'POST':
        selected_column = request.form.get('column_name')
        if selected_column in df.columns:
            column_data = df[selected_column]
            # Calculate various statistics
            stats_df = pd.DataFrame({
                'Count': [column_data.count()],
                'Mean': [column_data.mean()],
                'Median': [column_data.median()],
                'Standard Deviation': [column_data.std()],
                'Variance': [column_data.var()],
                'Minimum': [column_data.min()],
                'Maximum': [column_data.max()],
                '25th Percentile': [column_data.quantile(0.25)],
                '50th Percentile': [column_data.quantile(0.50)],
                '75th Percentile': [column_data.quantile(0.75)],
                'Skewness': [column_data.skew()],
                'Kurtosis': [column_data.kurtosis()]
            })

            # Convert the DataFrame to HTML for rendering
            stats = stats_df.to_html(classes='table table-striped', index=False)
            
            # Generate Histogram
            fig, ax = plt.subplots()
            sns.histplot(column_data, kde=False, ax=ax)
            ax.set_title('Histogram of ' + selected_column)
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Frequency')
            hist_image = io.BytesIO()
            plt.savefig(hist_image, format='png')
            plt.close(fig)
            hist_image.seek(0)
            hist_url = base64.b64encode(hist_image.getvalue()).decode('utf-8')
            
            # Generate Box Plot
            fig, ax = plt.subplots()
            sns.boxplot(x=column_data, ax=ax)
            ax.set_title('Box Plot of ' + selected_column)
            ax.set_xlabel(selected_column)
            box_image = io.BytesIO()
            plt.savefig(box_image, format='png')
            plt.close(fig)
            box_image.seek(0)
            box_url = base64.b64encode(box_image.getvalue()).decode('utf-8')
            
            # Generate Density Plot
            fig, ax = plt.subplots()
            sns.kdeplot(column_data, ax=ax)
            ax.set_title('Density Plot of ' + selected_column)
            ax.set_xlabel(selected_column)
            density_image = io.BytesIO()
            plt.savefig(density_image, format='png')
            plt.close(fig)
            density_image.seek(0)
            density_url = base64.b64encode(density_image.getvalue()).decode('utf-8')
        else:
            stats = "<p class='text-danger'>Invalid column name selected.</p>"

    return render_template(
        'calculate_stats.html',
        stats=stats,
        filename=filename,
        columns=columns,
        hist_url='data:image/png;base64,' + hist_url if hist_url else None,
        box_url='data:image/png;base64,' + box_url if box_url else None,
        density_url='data:image/png;base64,' + density_url if density_url else None
    )
@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    cleaning_steps = []
    cleaned_data = None

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'drop_na':
            df_cleaned = df.dropna()
            cleaning_steps.append('Dropped rows with missing values.')
        elif action == 'fill_na':
            fill_methods = request.form.getlist('fill_method')
            for method in fill_methods:
                if method == 'mean':
                    mean_value = df.mean(numeric_only=True)
                    df.fillna(mean_value, inplace=True)
                    cleaning_steps.append('Filled missing values with column means.')
                elif method == 'median':
                    median_value = df.median(numeric_only=True)
                    df.fillna(median_value, inplace=True)
                    cleaning_steps.append('Filled missing values with column medians.')
                elif method == 'mode':
                    mode_value = df.mode().iloc[0]
                    df.fillna(mode_value, inplace=True)
                    cleaning_steps.append('Filled missing values with column modes.')
                elif method == 'zeros':
                    df.fillna(0, inplace=True)
                    cleaning_steps.append('Filled missing values with zeros.')
                elif method == 'custom':
                    custom_value = request.form.get('custom_value', '0')
                    df.fillna(custom_value, inplace=True)
                    cleaning_steps.append(f'Filled missing values with custom value "{custom_value}".')
                elif method == 'interpolate':
                    df.interpolate(method='linear', inplace=True)
                    cleaning_steps.append('Interpolated missing values.')
        elif action == 'drop_duplicates':
            df_cleaned = df.drop_duplicates()
            cleaning_steps.append('Dropped duplicate rows.')
        elif action == 'trim_whitespace':
            df_cleaned = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            cleaning_steps.append('Trimmed whitespace from all string values.')

        # Save the cleaned data
        cleaned_filename = 'cleaned_' + filename
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)

        # Display the cleaned data
        cleaned_data = df.head().to_html(classes='table table-striped', index=False)

        return render_template('data_cleaning.html', cleaned_data=cleaned_data, filename=filename, steps=cleaning_steps, download_link=url_for('download_file', filename=cleaned_filename))

    return render_template('data_cleaning.html', cleaned_data=None, filename=filename, steps=cleaning_steps)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
