import pandas as pd
import urduhack
import streamlit as st
import pickle



loaded_model = pickle.load(open("traind_LR_classifier.pkle", "rb"))

# using streamlet library write a code in which a user submite an urdu coment and get prediction



# Define the Streamlit app
def main():
  st.title('Urdu Sentiment Prediction')

  # Create a text area for user input
  user_input = st.text_area('Enter your Urdu comment:', height=100)

  # Create a button for prediction
  if st.button('Predict'):
    # Preprocess the user input
    normalized_text = normalize(user_input)
    punc_removed = remove_punctuation(normalized_text)
    accent_removed = remove_accents(punc_removed)
    url_replaced = replace_urls(accent_removed)
    email_replaced = replace_emails(url_replaced)
    number_replaced = replace_numbers(email_replaced)
    currency_replaced = replace_currency_symbols(number_replaced)
    english_removed = remove_english_alphabets(currency_replaced)
    whitespace_normalized = normalize_whitespace(english_removed)
    stopwords_removed = remove_stopwords(whitespace_normalized)
    lemmatized_text = lemitizeStr(stopwords_removed)

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



