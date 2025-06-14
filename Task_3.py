import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Step 1: Load Plain Text File (.ft.txt) 
def load_ft_file(file_path, limit=100):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines[:limit]:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            label = "Positive" if "__label__2" in parts[0] else "Negative"
            review = parts[1]
            data.append((review, label))
    return pd.DataFrame(data, columns=["review", "label"])

#  Set your actual file path here
file_path = r"C:\Users\Eric Kinyanjui\Desktop\AI 4 S.E\AI-Project-3\test.ft.txt"
df = load_ft_file(file_path, limit=100)  # You can increase limit if needed

#  Step 2: Setup spaCy and Product Matcher 
nlp = spacy.load("en_core_web_sm")

# Product/brand keywords to look for (add more if needed)
product_keywords = [
    "iPhone", "Samsung", "Kindle", "Echo", "Fire Stick", "Sony",
    "Apple", "Bose", "Lenovo", "HP", "laptop", "headphones", "Amazon"
]
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in product_keywords]
matcher.add("PRODUCT", patterns)

# Rule-based sentiment keywords
positive_words = {"amazing", "great", "love", "perfect", "fantastic", "awesome", "excellent", "fast", "good"}
negative_words = {"terrible", "broke", "worst", "poor", "disappointed", "bad", "slow", "awful"}

#  Step 3: Analyze Review
def analyze_review(text):
    doc = nlp(text)

    # Product/entity matcher
    matches = matcher(doc)
    entities = list(set([doc[start:end].text for _, start, end in matches]))

    # Simple rule-based sentiment
    sentiment_score = sum(1 for token in doc if token.lemma_.lower() in positive_words) - \
                      sum(1 for token in doc if token.lemma_.lower() in negative_words)
    
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    return entities, sentiment

# Step 4: Apply to Reviews 
print("\n=== Sample Analysis ===\n")
for i in range(10):  # Adjust number of reviews to display
    review_text = df.loc[i, "review"]
    original_label = df.loc[i, "label"]
    entities, sentiment = analyze_review(review_text)

    print(f"Review: {review_text}")
    print(f"Extracted Entities: {entities}")
    print(f"Rule-Based Sentiment: {sentiment}")
    print(f"Original Dataset Label: {original_label}")
    print("-" * 60)
