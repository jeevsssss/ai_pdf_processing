from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from pdf2image import convert_from_path
from pytesseract import image_to_string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from summarizer import Summarizer
from keybert import KeyBERT
import os
import spacy
import logging
import re
import json
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Set up database folder
DATABASE_FOLDER = 'database'
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)  # Ensure the database directory exists

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Set the logging level for transformers library
logging.getLogger("transformers").setLevel(logging.WARNING)

# Directory to store analysis results as JSON files
ANALYSIS_RESULTS_DIR = 'analysis_results'
os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)

# Initialize lazy-loaded models and libraries
bert_summarizer = None
keybert_model = None
summarization_tokenizer = None
summarization_model = None
token_classifier = None
distilled_student_sentiment_classifier = None

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize BERT Summarizer (lazy-loaded)
def get_bert_summarizer():
    global bert_summarizer
    if bert_summarizer is None:
        bert_summarizer = Summarizer()
    return bert_summarizer

# Initialize KeyBERT model (lazy-loaded)
def get_keybert_model():
    global keybert_model
    if keybert_model is None:
        keybert_model = KeyBERT()
    return keybert_model

# Load the token classification pipeline (lazy-loaded)
def get_token_classifier():
    global token_classifier
    if token_classifier is None:
        model_checkpoint = "xlm-roberta-large-finetuned-conll03-english"
        token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")
    return token_classifier

# Load the sentiment classifier pipeline (lazy-loaded)
def get_sentiment_classifier():
    global distilled_student_sentiment_classifier
    if distilled_student_sentiment_classifier is None:
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None
        )
    return distilled_student_sentiment_classifier

# Load the BART Summarization model (lazy-loaded)
def get_summarization_model():
    global summarization_tokenizer, summarization_model
    if summarization_tokenizer is None:
        summarization_model_checkpoint = "facebook/bart-large-cnn"
        summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_checkpoint)
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_checkpoint)
    return summarization_tokenizer, summarization_model

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

    # Check if analysis results JSON exists
    analysis_json_path = os.path.join(ANALYSIS_RESULTS_DIR, f'{selected_pdf}.json')
    if os.path.exists(analysis_json_path):
        # Load analysis results from JSON
        analysis_results = load_analysis_results(analysis_json_path)
    else:
        # Get text from the entire PDF
        pdf_text = get_text_from_any_pdf(pdf_path)

        # Preprocess text (remove citations, references, journal names)
        preprocessed_text = preprocess_text(pdf_text)

        # Generate summary using BERT and then BART
        summary = generate_summary(preprocessed_text)

        # Find negative feedback about the product
        negative_feedback = analyze_feedback(preprocessed_text)

        # Extract keywords using KeyBERT
        keywords = extract_keywords(preprocessed_text)

        # Extract organizations
        organizations = extract_organizations(pdf_text)

        # Prepare analysis results dictionary
        analysis_results = {
            'summary': summary,
            'negative_feedback': negative_feedback,
            'keywords': keywords,
            'organizations': organizations
        }

        # Save analysis results to JSON
        save_analysis_results(analysis_results, analysis_json_path)

    return render_template('analysis.html', **analysis_results, selected_pdf=selected_pdf)

@app.route('/rerun_analysis', methods=['POST'])
def rerun_analysis():
    selected_pdf = request.json.get('selected_pdf')
    pdf_path = os.path.join(app.config['DATABASE_FOLDER'], selected_pdf)
    analysis_json_path = os.path.join(ANALYSIS_RESULTS_DIR, f'{selected_pdf}.json')

    if os.path.exists(pdf_path):
        # Rerun analysis functions and update the JSON file
        # Add your analysis functions here and update the JSON file accordingly

        # Get text from the entire PDF
        pdf_text = get_text_from_any_pdf(pdf_path)

        # Preprocess text (remove citations, references, journal names)
        preprocessed_text = preprocess_text(pdf_text)

        # Generate summary using BERT and then BART
        summary = generate_summary(preprocessed_text)

        # Find negative feedback about the product
        negative_feedback = analyze_feedback(preprocessed_text)

        # Extract keywords using KeyBERT
        keywords = extract_keywords(preprocessed_text)

        # Extract organizations
        organizations = extract_organizations(pdf_text)

        # Prepare analysis results dictionary
        analysis_results = {
            'summary': summary,
            'negative_feedback': negative_feedback,
            'keywords': keywords,
            'organizations': organizations
        }

        # Save updated analysis results to JSON
        save_analysis_results(analysis_results, analysis_json_path)

        return jsonify({'message': 'Analysis re-run successfully.'}), 200
    else:
        return jsonify({'error': 'PDF file not found.'}), 404

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

def preprocess_text(text):
    journal_names = ["IEEE Trans.", "ACM", "APA", "MLA", "Artif. Intell.", "Ai Soc.", "Mind", "Scientometrics"]
    journal_abbreviations = ["IEEE", "ACM", "APA", "MLA", "AI", "AIS", "Mind", "Sci."]
    reference_keywords = ["vol.", "no.", "pp.", "Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.",
                          "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
    all_journal_keywords = journal_names + journal_abbreviations

    sentences = text.split('\n')

    abstract_found = False
    cleaned_sentences = []
    for sentence in sentences:
        if "abstract" in sentence.lower():
            abstract_found = True
        if abstract_found:
            if not any(keyword in sentence for keyword in all_journal_keywords) and \
               not any(keyword in sentence for keyword in reference_keywords):
                sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)
                cleaned_sentences.append(sentence)

    cleaned_text = '\n'.join(cleaned_sentences)
    return cleaned_text

def generate_summary(text):
    summarization_tokenizer, summarization_model = get_summarization_model()
    inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=150, min_length=125, length_penalty=2.0, num_beams=8, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def analyze_feedback(text):
    meaningful_sentences = filter_meaningful_sentences(text)
    feedback_with_scores = []

    # Analyze sentiment for each sentence and store it with its sentiment score
    for sentence in meaningful_sentences:
        # Truncate the sentence if it exceeds the maximum token length
        truncated_sentence = sentence[:512]

        # Analyze sentiment for the truncated sentence
        feedback = get_sentiment_classifier()(truncated_sentence)

        for result in feedback:
            for res in result:
                if res['label'] == 'negative':
                    feedback_with_scores.append((truncated_sentence, res['score']))

    # Sort the feedback sentences based on their sentiment scores (higher scores are more negative)
    sorted_feedback = sorted(feedback_with_scores, key=lambda x: x[1], reverse=True)

    # Extract the top 5 most negative feedback sentences
    top_5_negative_feedback = [item[0] for item in sorted_feedback[:5]]

    return top_5_negative_feedback

def filter_meaningful_sentences(text):
    meaningful_sentences = []
    doc = nlp(text)
    for sentence in doc.sents:
        if any(token.pos_ in {'NOUN', 'VERB', 'ADJ'} for token in sentence):
            meaningful_sentences.append(sentence.text)
    return meaningful_sentences

def extract_keywords(text):
    keybert_model = get_keybert_model()
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return ', '.join([keyword[0] for keyword in keywords])

def extract_organizations(doc):
    token_classifier = get_token_classifier()
    classifier_output = token_classifier(doc)
    org_names = [item['word'] for item in classifier_output if item['entity_group'] == 'ORG']
    org_names = list(set(org_names))
    return org_names

def get_uploaded_pdfs():
    return [file for file in os.listdir(app.config['DATABASE_FOLDER']) if file.endswith('.pdf')]

def load_analysis_results(json_file):
    with open(json_file, 'r') as f:
        analysis_results = json.load(f)
    return analysis_results

def save_analysis_results(analysis_results, json_file):
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=4)

if __name__ == '__main__':
    app.run(debug=True)

       
