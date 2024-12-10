from flask import Flask, render_template, request, jsonify, session
import random
import nltk
import pandas as pd
from flask_session import Session
from spellchecker import SpellChecker  
from sklearn_crfsuite import CRF 
from nltk import pos_tag, word_tokenize
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

spell = SpellChecker()

# Load typo correction dataset
typo_data = pd.read_excel('modified_dataset4.xlsx')

typo_data['Incorrect'] = typo_data['incorrect'].astype(str)
typo_data['Correct'] = typo_data['correct'].astype(str)

typo_data = typo_data.sample(n=2000, random_state=42)

incorrect_words = typo_data['Incorrect'].tolist()
correct_words = typo_data['Correct'].tolist()

X_train = [[{'word': word}] for word in incorrect_words]  
y_train = [[correct_word] for correct_word in correct_words]

crf = CRF(algorithm='lbfgs', max_iterations=10)  
crf.fit(X_train, y_train)

# Load reading, listening, and writing data
reading_data = pd.read_excel('static/questions/reading.xlsx')
listening_data = pd.read_excel('static/questions/listening.xlsx')
writing_data = pd.read_excel('static/questions/writing.xlsx')


reading_questions = [
    {
        "text": row['text'] if 'text' in row else "",  # Include the text column if available
        "question": row['question'],
        "choices": [choice for choice in [row['choice a'], row['choice b'], row['choice c'], row['choice d']] if choice != "-"],
        "answer": row['answer']
    }
    for _, row in reading_data.iterrows()
]


# Process listening data
listening_questions = [
    {
        "question": row['question'],
        "choices": [
            row['choice a'], 
            row['choice b'], 
            row['choice c']
        ] + (
            [row['choice d']] if pd.notna(row['choice d']) and row['choice d'] != '-' else []
        ) + (
            [row['choice e']] if pd.notna(row['choice e']) and row['choice e'] != '-' else []
        ),
        "answer": row['answer'],
        "audio": row['audio'] if pd.notna(row['audio']) else None  # Include audio if available
    }
    for _, row in listening_data.iterrows()
]

# Process writing data
writing_questions = [
    {"prompt": row['question']} for _, row in writing_data.iterrows()
]

# Update questions dictionary
questions = {
    'reading': reading_questions,
    'listening': listening_questions,
    'writing': writing_questions
}

@app.route('/')
def home():
    session.clear()  
    return render_template('index.html')

@app.route('/submit_message', methods=['POST'])
def submit_message():
    user_message = request.form.get('message', '').strip().lower()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'current_section' not in session:
        session['current_section'] = None
    if 'current_question' not in session:
        session['current_question'] = None

    chat_history = session['chat_history']
    current_section = session['current_section']
    current_question = session['current_question']

    chat_history.append({'sender': 'user', 'message': user_message})

    if user_message == "stop":
        session['current_section'] = None
        session['current_question'] = None
        bot_message = "You've stopped the current section. Please choose a section: Reading, Listening, or Writing."
        chat_history.append({'sender': 'bot', 'message': bot_message})
        session['chat_history'] = chat_history
        return jsonify({'chat_history': chat_history})

    if not current_section:
        section = user_message
        if section in questions:
            session['current_section'] = section
            current_section = section
            current_question = random.choice(questions[section])
            session['current_question'] = current_question
            bot_message = get_bot_message(current_question, current_section)
            chat_history.append({'sender': 'bot', 'message': bot_message})
        else:
            bot_message = "Please choose a valid section: Reading, Listening, or Writing."
            chat_history.append({'sender': 'bot', 'message': bot_message})
    else:
        if current_section in ['reading', 'listening']:
            correct_answer = current_question['answer']
            if user_message.upper() == correct_answer:
                bot_message = "Correct!"
                status = 'correct'
            else:
                bot_message = f"Incorrect. The correct answer is {correct_answer}."
                status = 'incorrect'
            chat_history.append({'sender': 'bot', 'message': bot_message, 'status': status})

            current_question = random.choice(questions[current_section])
            session['current_question'] = current_question
            bot_message = get_bot_message(current_question, current_section)
            chat_history.append({'sender': 'bot', 'message': bot_message})
        elif current_section == 'writing':
            corrected_message = correct_typos(user_message)
            bot_message = corrected_message
            chat_history.append({'sender': 'bot', 'message': bot_message})

            current_question = random.choice(questions[current_section])
            session['current_question'] = current_question
            bot_message = get_bot_message(current_question, current_section)
            chat_history.append({'sender': 'bot', 'message': bot_message})

    session['chat_history'] = chat_history
    return jsonify({'chat_history': chat_history})

def get_bot_message(question_data, section):
    """Helper function to format bot's question message with choices displayed below."""
    if section == 'writing':
        return f"Writing Question: {question_data['prompt']}"
    elif section == 'reading':
        message = f"Text: {question_data['text']}<br><br>"  # Include the text before the question
        message += f"Question: {question_data['question']}<br>"
        for choice in question_data['choices']:
            message += f"{choice}<br>"
        return message
    else:
        message = f"Question: {question_data['question']}<br>"
        for choice in question_data['choices']:
            message += f"{choice}<br>"
        if section == 'listening' and question_data.get('audio'):
            audio_url = f"/static/{question_data['audio']}"  # Static path for Flask
            message += f"<br>Listen to the audio: <audio controls src='{audio_url}'></audio>"
        return message



def correct_typos(user_message):
    """Check for typos in the user's message using a spell checker, while retaining punctuation and removing spaces before punctuation."""
    tokens = nltk.word_tokenize(user_message) 
    corrected_tokens = []
    
    for token in tokens:
        if token.isalpha(): 
            corrected_word = spell.correction(token) or token
            corrected_tokens.append(corrected_word)
        else:
            corrected_tokens.append(token) 
    
    corrected_message = ' '.join(corrected_tokens)
    corrected_message = re.sub(r'\s+([?.!,])', r'\1', corrected_message)

    return corrected_message

if __name__ == '__main__':
    app.run(debug=True)
