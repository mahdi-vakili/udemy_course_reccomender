from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variables (initialized outside of functions for simplicity)
df = None
tfidf = None

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
    global df, tfidf
    
    df = read_csv_with_encodings(file_path)
    df = df.dropna(subset=['Title', 'Subtitle', 'Duration', 'Level', 'Image', 'Url', 'avg rate'])

    # Split the title based on the subtitle
    df['Title'] = df.apply(lambda row: row['Title'].split(row['Subtitle'])[0].strip(), axis=1)
    df['avg rate']=df.apply(lambda row: row['avg rate'].replace("Rating: ", "").replace("out of ", "/ "), axis=1)
    df['combined_features'] = (df['Title'] + " " + df['Subtitle']).str.lower()
    df['Duration'] = df['Duration'].apply(lambda x: float(x.split()[0]))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

def get_recommendations(keywords, Duration, Level=None):
    global df, tfidf
    
    keywords = [keyword.lower() for keyword in keywords]
    filtered_df = df[df['combined_features'].str.contains('|'.join(keywords), case=False, na=False)]
    filtered_df = filtered_df[filtered_df['Duration'] <= Duration]

    # Perform avg rate replacements early in the process
    filtered_df['new avg rate'] = filtered_df['avg rate'].replace("Rating: ", "").replace("out of ", "/")
    
    if Level and Level != "All Levels":
        filtered_df = filtered_df[(filtered_df['Level'].str.contains(Level, case=False, na=False)) |
                                  (filtered_df['Level'].str.contains("All Levels", case=False, na=False))]
    
    if filtered_df.empty:
        print("Filtered DataFrame is empty after applying filters.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches found
    
    filtered_tfidf_matrix = tfidf.transform(filtered_df['combined_features'])
    filtered_cosine_sim = cosine_similarity(filtered_tfidf_matrix, filtered_tfidf_matrix)
    
    sim_scores = list(enumerate(filtered_cosine_sim.mean(axis=1)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    course_indices = [i[0] for i in sim_scores]

    # Select only the necessary columns
    recommendations = filtered_df.iloc[course_indices][['Title', 'Subtitle', 'Duration', 'Level', 'Image', 'Url', 'new avg rate']]

    # Drop duplicates based on 'Title'
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
    Duration = float(request.form['Duration'])
    Level = request.form['Level']
    Level = Level if Level != "" else None
    
    recommendations = get_recommendations(keywords, Duration, Level)
    
    if recommendations.empty:
        return render_template('no_match.html')  # Render no_match.html for no recommendations found
    return render_template('result.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    preprocess_data('udemy_courses.csv')  # Load and preprocess data before running the app
    app.run(debug=True)
