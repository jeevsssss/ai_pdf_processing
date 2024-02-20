from flask import Flask, render_template, request, flash, redirect, url_for
from pdf2image import convert_from_path
from pytesseract import image_to_string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from summarizer import Summarizer
from keybert import KeyBERT
import os
import spacy
import logging
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

DATABASE_FOLDER = 'database'
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

# Ensure the database directory exists
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize BERT Summarizer
bert_summarizer = Summarizer()

# Initialize KeyBERT model
keybert_model = KeyBERT()

# Set the logging level for transformers library
logging.getLogger("transformers").setLevel(logging.WARNING)

# Load the token classification pipeline
model_checkpoint = "xlm-roberta-large-finetuned-conll03-english"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)

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

    pdf_files = get_uploaded_pdfs()
    return render_template('list.html', pdf_files=pdf_files)

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
    
@app.route('/list_pdfs')
def list_pdfs():
    pdf_files = get_uploaded_pdfs()
    return render_template('list.html', pdf_files=pdf_files)



@app.route('/analyze', methods=['POST'])
def analyze():
    selected_pdf = request.form['selected_pdf']
    pdf_path = os.path.join(app.config['DATABASE_FOLDER'], selected_pdf)

    if not os.path.exists(pdf_path):
        flash(f"PDF file '{selected_pdf}' not found.", 'error')
        return redirect(url_for('index'))

    # Get text from the entire PDF
    pdf_text = get_text_from_any_pdf(pdf_path)

    # Preprocess text (remove citations, references, journal names)
    preprocessed_text = preprocess_text(pdf_text)

    # Generate summary using BERT and then BART
    summary = generate_summary(preprocessed_text)

    # Find negative and positive feedback about the product
    negative_feedback, positive_feedback = analyze_feedback(preprocessed_text)

    # Extract keywords using KeyBERT
    keywords = extract_keywords(preprocessed_text)

    # Extract organizations
    organizations = extract_organizations(pdf_text)

    return render_template('analysis.html', summary=summary,
                           positive_feedback=positive_feedback[:5], negative_feedback=negative_feedback[:5],
                           keywords=keywords, organizations=organizations)



# Function to extract text from any PDF
# def get_text_from_first_3_pages(pdf_file):
#     try:
#         images = convert_from_path(pdf_file, first_page=1, last_page=3)
#         final_text = ""
#         for pg, img in enumerate(images):
#             final_text += image_to_string(img)
#         return final_text
#     except Exception as e:
#         flash(f"Error extracting text from PDF: {str(e)}", 'error')
#         return ""

# Function to preprocess text
def preprocess_text(text):
    # Define common journal names and their abbreviations
    journal_names = ["IEEE Trans.", "ACM", "APA", "MLA", "Artif. Intell.", "Ai Soc.", "Mind", "Scientometrics"]
    journal_abbreviations = ["IEEE", "ACM", "APA", "MLA", "AI", "AIS", "Mind", "Sci."]

    # Define keywords indicative of reference entries
    reference_keywords = ["vol.", "no.", "pp.", "Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", 
                         "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]

    # Combine journal names and abbreviations
    all_journal_keywords = journal_names + journal_abbreviations

    # Split the text into sentences
    sentences = text.split('\n')  # You may need to adjust this based on how sentences are separated in your PDF

    # Flag to indicate if "Abstract" is found
    abstract_found = False

    # Filter out sentences preceding the word "Abstract"
    cleaned_sentences = []
    for sentence in sentences:
        # Check if the sentence contains the word "Abstract"
        if "abstract" in sentence.lower():
            abstract_found = True

        # If "Abstract" is found, start filtering out sentences
        if abstract_found:
            # Check if the sentence contains any of the journal names, abbreviations, or reference keywords
            if not any(keyword in sentence for keyword in all_journal_keywords) and \
               not any(keyword in sentence for keyword in reference_keywords):
                # Remove links
                sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)
                cleaned_sentences.append(sentence)

    # Join the cleaned sentences back into a single string
    cleaned_text = '\n'.join(cleaned_sentences)

    return cleaned_text

# Function to generate summary using BERT
def generate_summary(text):
    summary = bert_summarizer(text, min_length=100, max_length=250)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    final_summary = summarizer(summary, max_length=130, min_length=100, do_sample=False)
    return final_summary[0]['summary_text']

# Function to analyze feedback about a specific product
def analyze_feedback(text):
    meaningful_sentences = filter_meaningful_sentences(text)
    negative_feedback = []
    positive_feedback = []
    for sentence in meaningful_sentences:
        vs = analyzer.polarity_scores(sentence)
        if vs['compound'] < 0:  # Consider negative sentiment
            negative_feedback.append(sentence)
        elif vs['compound'] > 0:  # Consider positive sentiment
            positive_feedback.append(sentence)
    return negative_feedback, positive_feedback

# Function to filter out meaningless sentences
def filter_meaningful_sentences(text):
    meaningful_sentences = []
    doc = nlp(text)
    for sentence in doc.sents:
        # Check if the sentence contains meaningful content (e.g., nouns, verbs, adjectives)
        if any(token.pos_ in {'NOUN', 'VERB', 'ADJ'} for token in sentence):
            meaningful_sentences.append(sentence.text)
    return meaningful_sentences

# Function to extract keywords using KeyBERT
def extract_keywords(text):
    # Extract keywords using KeyBERT
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return ', '.join([keyword[0] for keyword in keywords])

# Function to extract organizations using token classification
def extract_organizations(doc):
    # Perform token classification
    classifier_output = token_classifier(doc)

    # Extract organization names (ORG entities)
    org_names = [item['word'] for item in classifier_output if item['entity_group'] == 'ORG']

    # Remove duplicates
    org_names = list(set(org_names))

    return org_names


# Function to list uploaded PDF files
def get_uploaded_pdfs():
    return [file for file in os.listdir(app.config['DATABASE_FOLDER']) if file.endswith('.pdf')]

import pdfkit

# This function generates a PDF from the HTML content
def generate_pdf(html_content, output_path):
    pdfkit.from_string(html_content, output_path)


if __name__ == '__main__':
    app.run(debug=True)
