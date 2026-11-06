import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Part 1: rule-based replies (keywords)
RULES = {
    r"\b(hi|hello|hey)\b": "Hello! How can I help you?",
    r"\b(bye|goodbye)\b": "Goodbye! Have a great day.",
    r"\bthanks?\b|thank you": "You're welcome!",
}

# Part 2: semantic FAQ (meaning-based matching)
FAQ = {
    "How do I reset my password?": "Click 'Forgot password' on the login page and follow the email link.",
    "What are your working hours?": "We are available Monday to Friday, 9 AM to 6 PM.",
    "How can I contact support?": "Email support@example.com or use the website chat.",
    "How do I track my order?": "Open 'My Orders' and click Track next to your order.",
    "What is your refund policy?": "Refunds are available within 7 days of purchase.",
}

questions = list(FAQ.keys())
vectorizer = TfidfVectorizer(stop_words="english")
faq_vectors = vectorizer.fit_transform(questions)
THRESHOLD = 0.25

def rule_reply(text):
    for pattern, reply in RULES.items():
        if re.search(pattern, text.lower()):
            return reply
    return None

def semantic_reply(text):
    scores = cosine_similarity(vectorizer.transform([text]), faq_vectors)[0]
    best = scores.argmax()
    if scores[best] >= THRESHOLD:
        return FAQ[questions[best]]
    return None

def respond(text):
    text = text.strip()
    if not text:
        return "Please type something."
    return (rule_reply(text) or semantic_reply(text)
            or "Sorry, I didn't understand. Please contact support.")

if __name__ == "__main__":
    print("Chatbot ready. Type 'bye' to exit.")
    while True:
        user = input("You: ")
        print("Bot:", respond(user))
        if re.search(r"\b(bye|goodbye)\b", user.lower()):
            break