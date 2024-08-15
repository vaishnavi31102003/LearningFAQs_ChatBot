import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample FAQs (Question and Answer pairs)
faqs = [
    {"question": "Hello",
     "answer": "Hello, welcome! How can we assist you with your educational needs today?"},
    {"question": "What are the available courses?",
     "answer": "We offer a wide range of courses across various fields such as technology, arts, sciences, and business. You can explore them on our website."},
    {"question": "How can I apply for a course?",
     "answer": "You can apply for a course by visiting the course page and clicking on the 'Apply Now' button. Follow the instructions provided."},
    {"question": "Do you offer scholarships?",
     "answer": "Yes, we offer scholarships for deserving students. You can find more details on the scholarships page on our website."},
    {"question": "How can I contact academic support?",
     "answer": "You can contact academic support by emailing academic_support@example.com or by visiting the support section on our platform."},
    {"question": "What are the payment options for course fees?",
     "answer": "We accept payments via credit cards, debit cards, net banking, and PayPal. Payment plans are also available for select courses."},
]



# Preprocessing functions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if
              word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)


# Preprocess FAQs
faq_questions = [preprocess_text(faq['question']) for faq in faqs]


# Function to find the most similar FAQ
def find_most_similar_faq(user_input, faqs, faq_questions):
    # Preprocess user input
    user_input_processed = preprocess_text(user_input)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(faq_questions + [user_input_processed])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1])

    # Find the index of the most similar FAQ
    similar_faq_index = np.argmax(cosine_similarities)
    return faqs[similar_faq_index], cosine_similarities[0][similar_faq_index]


# Main chatbot loop
def faq_chatbot():
    print("Welcome to the FAQ chatbot! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a nice day!")
            break

        faq, similarity = find_most_similar_faq(user_input, faqs, faq_questions)

        # If similarity is above a certain threshold, provide the answer
        if similarity > 0.2:  # Adjust threshold as needed
            print(f"Chatbot: {faq['answer']}")
        else:
            print("Chatbot: I'm sorry, I don't understand your question. Can you please rephrase?")


# Run the chatbot
faq_chatbot()