from transformers import AutoTokenizer

# Load the tokenizer (Example: BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Your text
text = """Em EES en el TL TT - NON-ST ABLIZED ATTACHED RIDER ar THE APARTMENT IS... (your full text here)"""

# Tokenize the text
tokens = tokenizer(text, truncation=False)  # No truncation to check length

# Print token length
print(len(tokens["input_ids"]))

# Check if it exceeds 512
if len(tokens["input_ids"]) > 512:
    print("The text exceeds 512 tokens after tokenization.")
else:
    print("The text is within the 512 token limit.")
