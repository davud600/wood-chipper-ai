import re

from autocorrect import Speller

from config.settings import DOCUMENT_TYPES


spell = Speller(lang="en")


def get_formatted_content_batch(curr_page_idx: int, contents: list[str]) -> str:
    content_batch = ""

    for i, content in enumerate(contents):
        if curr_page_idx == i:
            content_batch += f"<curr_page>{content}</curr_page>"
        elif i < curr_page_idx:
            content_batch += f"<prev_page_{curr_page_idx - i}>{content}</prev_page_{curr_page_idx - i}>"
        elif i > curr_page_idx:
            content_batch += f"<next_page_{i - curr_page_idx}>{content}</next_page_{i - curr_page_idx}>"

    return content_batch


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


def filter_junky_lines(text: str) -> str:
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        tokens = line.split()
        if (
            len(tokens) <= 6
            and sum(1 for t in tokens if re.fullmatch(r"[A-Z0-9]{2,}", t))
            >= len(tokens) - 1
            and not any(c.islower() for c in line)
        ):
            continue  # Junk — skip
        clean_lines.append(line)
    return "\n".join(clean_lines)


def clean_text(text: str) -> str:
    # Normalize unicode (e.g., smart quotes → straight quotes)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove overly long garbage tokens (e.g., "_____", "======", "|||||", etc.)
    text = re.sub(r"([=\-_\|]{3,})", " ", text)

    # Remove isolated symbols or sequences of non-word characters
    text = re.sub(r"\b[^a-zA-Z0-9\s]{2,}\b", " ", text)

    # Remove bracketed noise (e.g., [cid:image001.jpg@...], [Page 1])
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove sequences of digits with no alphabet around (e.g., timestamps, hex, table IDs)
    text = re.sub(r"\b\d{4,}\b", " ", text)

    # Remove single characters that are surrounded by spaces (unless whitelisted)
    whitelist = {
        "a",
        "i",
        "A",
        "I",
        "U",
        "S",
        "T",
        "F",
        "G",
    }  # whitelist common important letters
    text = " ".join(
        [word for word in text.split() if len(word) > 1 or word in whitelist]
    )

    text = filter_junky_lines(text)

    # Collapse multiple spaces, line breaks, tabs
    text = re.sub(r"\s+", " ", text).strip()

    # Final cleaning: only keep useful punctuation
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


def parse_formatted_content_batch_to_sections(
    content_batch: str,
) -> tuple[str, str, str]:
    pattern = re.compile(r"<(prev_page_(\d+)|curr_page|next_page_(\d+))>(.*?)</\1>")

    prev_pages = {}
    next_pages = {}
    curr_content = ""

    for match in pattern.finditer(content_batch):
        tag = match.group(1)
        prev_offset = match.group(2)
        next_offset = match.group(3)
        content = match.group(4)

        if tag == "curr_page":
            curr_content = content
        elif prev_offset is not None:
            idx = int(prev_offset)
            prev_pages[idx] = content
        elif next_offset is not None:
            idx = int(next_offset)
            next_pages[idx] = content

    # Sort by offset (e.g. prev_page_3, prev_page_2, prev_page_1 → ordered as 3, 2, 1 → combine in that order)
    prev_combined = "".join(
        prev_pages[i] for i in sorted(prev_pages.keys(), reverse=True)
    )
    next_combined = "".join(next_pages[i] for i in sorted(next_pages.keys()))

    return prev_combined, curr_content, next_combined
