import spacy


class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return list({ent.text for ent in doc.ents})
