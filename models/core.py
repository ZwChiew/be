import inflect
import re
import spacy
from googlesearch import search
from autocorrect import Speller

# Create an inflect engine
p = inflect.engine()
nlp = spacy.load("en_core_web_sm")
spell = Speller(lang='en')


def search_website(keyword):
    # Perform the Google search
    search_results = search(keyword, lang="en", stop=1)
    # Extract and return the first website link
    for result in search_results:
        s = (
            "Thank you for your question, but the query you've provided is not addressed in the context information"
            "(Strata Management Act). Please only ask question related to Strata Council Matters. \n"
            f"For more detailed information on topics beyond the Strata Act, you may consult the local COB office or "
            f"refer to the following: {result}"
        )
        return s
    return "Sorry the context information (Strata Management Act) does not provide any information on this " \
           "matter. Please only ask question related to Strata Matters."
    # Return None if no results are found


# Define a function to convert digits to English words
def digits_to_words(match):
    return p.number_to_words(match.group())


def RAG_Preprocess(text):
    # Replace numeric digits with English words
    text = re.sub(r'\d+', digits_to_words, text)

    word_replacements = {'cob': 'Commissioner Of Building', 'jmb': 'Joint Management Body',
                         'jmc': 'Joint Management Subsidiary', 'smc': 'subsidiary management committee',
                         'smt': 'Strata Management Tribunal ', 'dlp': 'Defect Liability Period',
                         'mc': 'Management Corporation', '/': ' or ', 'agm': "Annual General Meeting",
                         'egm': "Extraordinary General Meeting"}
    for word, replacement in word_replacements.items():
        text = re.sub(r'\b' + word + r'\b', replacement,
                      text, flags=re.IGNORECASE)
    return text


# Define a function to perform the required transformations
def transform_text(text):
    # Replace numeric digits with English words
    text = re.sub(r'\d+', digits_to_words, text)

    word_replacements = {'cob': 'Commissioner Of Building', 'jmb': 'Joint Management Body',
                         'jmc': 'Joint Management Subsidiary', 'smc': 'subsidiary management committee',
                         'smt': 'Strata Management Tribunal ', 'dlp': 'Defect Liability Period',
                         'mc': 'Management Corporation', '/': ' or ', 'agm': "Annual General Meeting",
                         'egm': "Extraordinary General Meeting"}
    for word, replacement in word_replacements.items():
        text = re.sub(r'\b' + word + r'\b', replacement,
                      text, flags=re.IGNORECASE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Replace specific words
    return text


def custom_lemmatize(token):
    if token.text.lower() in ("building", "buildings"):
        return "building"  # Specify the lemma you want
    return token.lemma_


def clean_text(sentence):
    question = transform_text(sentence).lower()
    doc = nlp(question)
    lemmatized_sentence = ' '.join(
        [custom_lemmatize(token) for token in doc if not token.is_space])
    corrected_words = [spell(word) for word in lemmatized_sentence.split()]
    corrected_text = ' '.join(corrected_words)
    return corrected_text


def check_for_keywords(input_string):
    with open("dataset/error.txt", 'r') as file:
        keywords = [line.strip() for line in file]
    return any(keyword.lower() in input_string.lower() for keyword in keywords)
