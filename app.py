from flask import Flask, render_template, request, flash, redirect, url_for
from pdf2image import convert_from_path
from pytesseract import image_to_string
from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import spacy

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

DATABASE_FOLDER = 'database'
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

# Ensure the database directory exists
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

# Initialize KeyBERT model
kw_model = KeyBERT()

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file. Please choose a PDF file.', 'error')
            return redirect(url_for('index'))

        # Save the uploaded PDF file to the database folder
        pdf_path = os.path.join(app.config['DATABASE_FOLDER'], file.filename)
        file.save(pdf_path)

        flash('PDF file uploaded successfully.', 'success')
        return redirect(url_for('list_pdfs'))

@app.route('/list_pdfs')
def list_pdfs():
    pdf_files = get_uploaded_pdfs()
    return render_template('list.html', pdf_files=pdf_files)

@app.route('/analyze', methods=['POST'])
def analyze():
    selected_pdf = request.form['selected_pdf']
    product_name = request.form['product_name']

    pdf_path = os.path.join(app.config['DATABASE_FOLDER'], selected_pdf)

    if not os.path.exists(pdf_path):
        flash(f"PDF file '{selected_pdf}' not found.", 'error')
        return redirect(url_for('list_pdfs'))

    pdf_text = get_text_from_any_pdf(pdf_path)

    # Extract keywords from the PDF text
    keywords = extract_keywords(pdf_text)

    # Find negative feedback about the product
    negative_feedback = find_negative_feedback_about_product(pdf_text, product_name)

    return render_template('analysis.html', keywords=keywords, negative_feedback=negative_feedback, product_name=product_name)

import spacy

nlp = spacy.load("en_core_web_sm")

def filter_meaningless_sentences(text):
    meaningful_sentences = []
    doc = nlp(text)
    for sentence in doc.sents:
        # Check if the sentence contains meaningful content (e.g., nouns, verbs, adjectives)
        if any(token.pos_ in {'NOUN', 'VERB', 'ADJ'} for token in sentence):
            meaningful_sentences.append(sentence.text)
    return meaningful_sentences


# Function to extract text from any PDF
def get_text_from_any_pdf(pdf_file):
    try:
        images = convert_from_path(pdf_file)
        final_text = ""
        for pg, img in enumerate(images):
            final_text += image_to_string(img)
        return final_text
    except Exception as e:
        flash(f"Error extracting text from PDF: {str(e)}", 'error')
        return ""

# Function to extract keyword phrases using KeyBERT
def extract_keywords(doc):
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(2, 2))  # Extract 2-word phrases
    return [keyword[0] for keyword in keywords]

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

# Function to list uploaded PDF files
def get_uploaded_pdfs():
    return [file for file in os.listdir(app.config['DATABASE_FOLDER']) if file.endswith('.pdf')]

if __name__ == '__main__':
    app.run(debug=True)
