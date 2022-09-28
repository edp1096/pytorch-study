import spacy

spacy_en = spacy.load("en_core_web_sm")

en_text = "A Dog Run back corner near spare bedrooms"


def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]


print(tokenize(en_text))
