from flask import Flask, request, render_template
import pandas as pd
import re

# Global variables (initialized outside of functions for simplicity)
df = None

# Function to read CSV file with different encodings
def read_csv_with_encodings(file_path):
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Failed to read the CSV file with the given encodings")

# Read dataset and preprocess
def preprocess_data(file_path):
    global df
    
    df = read_csv_with_encodings(file_path)
    df = df.dropna(subset=['Title', 'Subtitle', 'Duration', 'Level', 'Image', 'Url', 'avg rate'])
    
    # Split the title based on the subtitle
    df['Title'] = df.apply(lambda row: row['Title'].split(row['Subtitle'])[0].strip(), axis=1)
    df['avg rate'] = df['avg rate'].apply(lambda x: x.replace("Rating: ", "").replace("out of ", "/"))
    df['combined_features'] = (df['Title'] + " " + df['Subtitle']).str.lower()
    df['Duration'] = df['Duration'].apply(lambda x: float(x.split()[0]))

def text_similarity(text1, text2):
    # Simple function to calculate cosine similarity between two texts
    text1 = re.sub(r'[^\w\s]', '', text1.lower())  # Remove punctuation and convert to lowercase
    text2 = re.sub(r'[^\w\s]', '', text2.lower())  # Remove punctuation and convert to lowercase
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = words1 & words2
    similarity = len(intersection) / float(len(words1) + len(words2) - len(intersection))
    
    return similarity

def get_recommendations(keywords, duration, level=None):
    global df
    
    keywords = [keyword.strip().lower() for keyword in keywords]
    recommendations = []
    
    for index, row in df.iterrows():
        if all(keyword in row['combined_features'] for keyword in keywords) and row['Duration'] <= duration:
            if level and level != "All Levels" and level.lower() not in row['Level'].lower():
                continue
            
            recommendations.append(row)
    
    recommendations = pd.DataFrame(recommendations)
    recommendations = recommendations[['Title', 'Subtitle', 'Duration', 'Level', 'Image', 'Url', 'avg rate']]
    recommendations = recommendations.drop_duplicates(subset=['Title'])
    
    return recommendations

# Flask application setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    keywords = request.form['keywords'].split(',')
    duration = float(request.form['Duration'])
    level = request.form['Level']
    level = level if level != "" else None
    
    recommendations = get_recommendations(keywords, duration, level)
    
    if recommendations.empty:
        return render_template('no_match.html')  # Render no_match.html for no recommendations found
    return render_template('result.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    preprocess_data('udemy_courses.csv')  # Load and preprocess data before running the app
    app.run(debug=True)
