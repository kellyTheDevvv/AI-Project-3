Part 3: Ethics & Optimization

Ethical Considerations

MNIST Handwritten Digits (Deep Learning Task):
  Potential Biases: Even with numbers, biases can creep in. If the handwriting in the dataset does not represent how everyone writes (e.g., only American styles), the model might struggle with different handwritings globally. Additionally, if some numbers have slightly more examples than others, the model may become better at recognizing those.
  How TensorFlow Fairness Indicators Help: This tool lets you check if your model performs differently across any defined group. For MNIST, you could hypothetically group digits by handwriting style. If the tool indicates that your model performs worse at certain styles, you have identified a bias.

Amazon Product Reviews (NLP Task):
  Potential Biases: This is where bias is more common. 
     Sentiment Bias: If reviews are mostly from a specific type of user, the model might not understand how others express themselves, e.g., sarcasm is tough for rule-based systems.
     Product Bias: If the reviews are all about electronics, your model might be great at finding tech brands but bad at finding clothing brands.
     Language Bias: If the model only sees English reviews, it will not understand other languages or even different English dialects.

How spaCy's Rule-Based Systems Help: You can fight bias directly with smart rules: 
   For Sarcasm: Instead of just looking for keywords, create rules that spot patterns like "positive word + negative context" (e.g., "amazing... it broke").
   For Different Products: Add specific rules to recognize brands or product names in categories your model struggles with.
   For Language: Develop different rule sets for various dialects or languages if your reviews cover them.
