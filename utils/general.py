import re

from autocorrect import Speller

from config.settings import DOCUMENT_TYPES


spell = Speller(lang="en")


def get_document_type(file_name: str) -> int:
    file_name = file_name.replace("-", " ").replace("_", " ").lower()

    if "proprietary" in file_name:
        return DOCUMENT_TYPES["proprietary-lease"]
    elif "tenant correspondence" in file_name:
        return DOCUMENT_TYPES["tenant-correspondence"]
    elif "transfer" in file_name and "title" in file_name.lower():
        return DOCUMENT_TYPES["transfer-of-title"]
    elif "purchase" in file_name:
        return DOCUMENT_TYPES["purchase-application"]
    elif "closing" in file_name:
        return DOCUMENT_TYPES["closing-document"]
    elif "alteration" in file_name:
        return DOCUMENT_TYPES["alteration-document"]
    elif "renovation" in file_name:
        return DOCUMENT_TYPES["renovation-document"]
    elif "refinance" in file_name:
        return DOCUMENT_TYPES["refinance-document"]
    elif "transfer" in file_name:
        return DOCUMENT_TYPES["transfer-document"]
    elif "sublease renewal" in file_name:
        return DOCUMENT_TYPES["sublease-renewal"]
    elif "sublease agreement" in file_name:
        return DOCUMENT_TYPES["sublease-agreement"]
    elif "sublease" in file_name:
        return DOCUMENT_TYPES["sublease"]
    elif "lease renewal" in file_name:
        return DOCUMENT_TYPES["lease-renewal"]
    elif "lease agreement" in file_name:
        return DOCUMENT_TYPES["lease-agreement"]
    elif "lease" in file_name:
        return DOCUMENT_TYPES["lease"]

    return DOCUMENT_TYPES["unknown"]


def clean_text(text: str) -> str:
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove extraneous punctuation
    text = re.sub(r'[^a-zA-Z0-9.,;:\'"\-\s]', "", text)
    return text


def is_safe_to_correct(english_words: set[str], word: str) -> bool:
    if len(word) <= 3:
        return False
    if word.lower() in english_words:
        return True
    if re.match(r"^[A-Z0-9]+$", word):
        return False
    if any(char.isdigit() for char in word):
        return False

    return True


def light_autocorrect(english_words: set[str], text: str) -> str:
    corrected_words = []

    for word in text.split():
        if is_safe_to_correct(english_words, word):
            corrected = spell(word)
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def split_into_n_chunks(lst, n):
    k, m = divmod(len(lst), n)

    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def split_arr(arr, size):
    return [arr[i : i + size] for i in range(0, len(arr), size)]
