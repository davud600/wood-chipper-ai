from typing import Dict, Any

from src.utils import DOCUMENT_TYPES

import openai
import json

openai.api_key = "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"

CommonDocumentsDataPointsPrompt = f"""
You are given OCR-extracted text from a document. Your task is to extract relevant data points and return a JSON object with the following structure:

{{
    "type": string | null,                // Type of document, some of the types include: {", ".join([type.replace('-', ' ').title() for type in list(DOCUMENT_TYPES.keys())[1:]])}
    "building": string | null,            // Building address of the property
    "unit": string | null,                // Unit or Apartment Number (usually contains 1-3 digits and 1-2 capitalized letters)
    "tenants": string | null,             // List of tenants as string seperated by ",". e.g. format: "John Snow,Filon Fisteku"
    "date": string | null,                // Date of the application (MUST be string, format: MM-DD-YYYY)
    "startDate": string | null,           // Date of agreement start (MUST be string, format: MM-DD-YYYY)
    "endDate": string | null,             // Date of agreement end (MUST be string, format: MM-DD-YYYY)
}}

Extract the above fields accurately from the following text and ENSURE THE RESPONSE IS JSON. If a field is not found in the text, return null for that field.
OCR Extracted Text:
"""


def clean_text(text: str) -> str:
    return text.replace("\x0c", "").strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_ai_content_to_js_obj(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def request_data_points(content: str) -> Dict[str, Any]:
    formatted_pages = normalize_whitespace(clean_text(content))
    prompt = f"{CommonDocumentsDataPointsPrompt}{formatted_pages}"

    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    content = (response.choices[0].message.content or "").strip()
    content = content.replace("`", "").replace("json", "")

    print("gpt response:\n", response.choices[0].message.content)

    return extract_ai_content_to_js_obj(content)
