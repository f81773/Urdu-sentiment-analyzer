import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import urduhack

loaded_model = pickle.load(open("traind_LR_classifier.pkle", "rb"))

def predict_sentiment(urdu_string):
    # Convert the Urdu string into a DataFrame
    df_new = pd.DataFrame({'review': [urdu_string]})

    # Apply UrduHack preprocessing
    df_new['review'] = df_new['review'].apply(normalize)
    df_new['review'] = df_new['review'].apply(remove_punctuation)
    df_new['review'] = df_new['review'].apply(remove_accents)
    df_new['review'] = df_new['review'].apply(replace_urls)
    df_new['review'] = df_new['review'].apply(replace_emails)
    df_new['review'] = df_new['review'].apply(replace_numbers)
    df_new['review'] = df_new['review'].apply(replace_currency_symbols)
    df_new['review'] = df_new['review'].apply(remove_english_alphabets)
    df_new['review'] = df_new['review'].apply(normalize_whitespace)

    # Remove stopwords
    df_new['review'] = df_new['review'].apply(remove_stopwords)

    # Lemmatize the text
    df_new['lemmatized_text'] = df_new['review'].apply(lemitizeStr)

    # Apply TF-IDF Vectorization
    new_test_vecs = TfidfVectorizer(max_features=max_feature_num, vocabulary=vectorizer.vocabulary_).fit_transform(df_new['lemmatized_text'])

    # Store the new vectorized text in a new variable
    new_text_vec = new_test_vecs
    prediction = loaded_model.predict(new_text_vec)

    if (prediction[0] == 0):
      return 'The coment is negative'
    else:
      return 'The coment is positive'


# Define the main function
def main():
    # Create a form to get input from the user
    with st.form("my_form"):
        text_input = st.text_area("Enter your Urdu text here:")
        submit_button = st.form_submit_button("Predict")

    # If the submit button is clicked, predict the sentiment
    if submit_button:
        prediction = predict_sentiment(text_input)
        st.write(prediction)

# Run the main function
if __name__ == "__main__":
    main()
