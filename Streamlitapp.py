import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import urduhack

# Function to preprocess text using urduhack
def preprocess_urdu_text(text):
    normalized_text = urduhack.normalize(text)
    punc_removed = urduhack.remove_punctuation(normalized_text)
    accent_removed = urduhack.remove_accents(punc_removed)
    url_replaced = urduhack.replace_urls(accent_removed)
    email_replaced = urduhack.replace_emails(url_replaced)
    number_replaced = urduhack.replace_numbers(email_replaced)
    currency_replaced = urduhack.replace_currency_symbols(number_replaced)
    english_removed = urduhack.remove_english_alphabets(currency_replaced)
    whitespace_normalized = urduhack.normalize_whitespace(english_removed)
    stopwords_removed = urduhack.remove_stopwords(whitespace_normalized)
    lemmatized_text = urduhack.lemitizeStr(stopwords_removed)
    return lemmatized_text

def load_model(file_path):
    try:
        # Load the trained model
        model = pickle.load(open(file_path, "rb"))
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def main():
    st.title('Urdu Sentiment Prediction')

    # Load the model
    loaded_model = load_model("trained_LR_classifier.pkle")
    if loaded_model is None:
        return

    # Create a text area for user input
    user_input = st.text_area('Enter your Urdu comment:', height=100)

    # Create a button for prediction
    if st.button('Predict'):
        # Preprocess the user input
        lemmatized_text = preprocess_urdu_text(user_input)

        # Vectorize the preprocessed text
        vectorizer = TfidfVectorizer(max_features=max_feature_num, vocabulary=your_vocabulary)
        new_test_vecs = vectorizer.fit_transform([lemmatized_text])

        # Predict the sentiment
        prediction = loaded_model.predict(new_test_vecs)[0]

        # Display the prediction
        st.write('The comment is positive.' if prediction == 1 else 'The comment is negative.')

# Run the Streamlit app
if __name__ == '__main__':
    max_feature_num = 1000  # Adjust this value based on your needs
    your_vocabulary = None  # Replace with your vocabulary or None to use the default
    main()
