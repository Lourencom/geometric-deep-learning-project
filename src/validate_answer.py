import string
import re

def get_first_sentence(text):
    """Extract the first sentence from text."""
    # Split by common sentence endings followed by space or newline
    sentences = re.split(r'[.!?][\s\n]', text)
    # Return the first non-empty sentence, or empty string if no sentences found
    return sentences[0].strip() if sentences else ""

def normalize_answer(s):
    # Lowercase, remove punctuation, and strip whitespace
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

def evaluate_answer(generated_text, expected_answer):
    return normalize_answer(expected_answer) in normalize_answer(generated_text)
