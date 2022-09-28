import nltk
from nltk.tokenize import word_tokenize


en_text = "A Dog Run back corner near spare bedrooms"

nltk.download("punkt")
print(word_tokenize(en_text))

print(en_text.split())
