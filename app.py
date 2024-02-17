# app.py
from flask import Flask, render_template, request
from pdf2image import convert_from_path
from pytesseract import image_to_string
from keybert import KeyBERT
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize KeyBERT model
kw_model = KeyBERT()

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from any PDF
def get_text_from_any_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    final_text = ""
    for pg, img in enumerate(images):
        final_text += image_to_string(img)
    return final_text

# Function to extract keyword phrases using KeyBERT
def extract_keywords(doc):
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2))  # Extract 2-word phrases
    return [keyword[0] for keyword in keywords]

# Function to filter meaningless sentences
def filter_meaningless_sentences(text):
    meaningful_sentences = []
    doc = nlp(text)
    for sentence in doc.sents:
        # Check if the sentence contains meaningful content (e.g., nouns, verbs, adjectives)
        if any(token.pos_ in {'NOUN', 'VERB', 'ADJ'} for token in sentence):
            meaningful_sentences.append(sentence.text)
    return meaningful_sentences

# Function to find negative feedback about a specific product
def find_negative_feedback_about_product(text, product_name):
    meaningful_sentences = filter_meaningless_sentences(text)
    negative_feedback = []
    for sentence in meaningful_sentences:
        if product_name.lower() in sentence.lower():
            vs = analyzer.polarity_scores(sentence)
            if vs['compound'] < 0:  # Consider negative sentiment
                negative_feedback.append(sentence)
    return negative_feedback

# Function to find overall negative feedback
def find_overall_negative_feedback(text):
    meaningful_sentences = filter_meaningless_sentences(text)
    negative_feedback = []
    for sentence in meaningful_sentences:
        vs = analyzer.polarity_scores(sentence)
        if vs['compound'] < 0:  # Consider negative sentiment
            negative_feedback.append(sentence)
    return negative_feedback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        product_name = request.form['product_name']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Save the uploaded PDF file
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(pdf_path)

        # Extract text from the uploaded PDF
        pdf_text = get_text_from_any_pdf(pdf_path)

        # Extract keyword phrases from the text
        keyword_phrases = extract_keywords(pdf_text)

        # Find negative feedback about the specific product
        negative_feedback_product = find_negative_feedback_about_product(pdf_text, product_name)

        # Find overall negative feedback
        overall_negative_feedback = find_overall_negative_feedback(pdf_text)

        return render_template('upload.html', keyword_phrases=keyword_phrases, 
                               negative_feedback_product=negative_feedback_product, 
                               overall_negative_feedback=overall_negative_feedback, 
                               product_name=product_name)

if __name__ == '__main__':
    app.run(debug=True)
