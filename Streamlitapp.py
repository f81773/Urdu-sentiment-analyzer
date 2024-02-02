# prompt: write the all above code in this cell

import pandas as pd
import urduhack
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("traind_LR_classifier.pkle", "rb"))

# Define the Streamlit app
def main():
    st.title('Urdu Sentiment Prediction')

    # Create a text area for user input
    user_input = st.text_area('Enter your Urdu comment:', height=100)

    # Create a button for prediction
    if st.button('Predict'):
        # Preprocess the user input
        normalized_text = urduhack.normalize(user_input)
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

        # Vectorize the preprocessed text
        new_test_vecs = TfidfVectorizer(max_features=max_feature_num, vocabulary=vectorizer.vocabulary_).fit_transform([lemmatized_text])

        # Predict the sentiment
        prediction = loaded_model.predict(new_test_vecs)[0]

        # Display the prediction
        if prediction == 0:
            st.write('The comment is negative.')
        else:
            st.write('The comment is positive.')

# Run the Streamlit app
if __name__ == '__main__':
    main()
