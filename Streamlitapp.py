import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import urduhack
urduhack.download()
from urduhack.normalization import normalize
from urduhack.preprocessing import normalize_whitespace, remove_punctuation, remove_accents, replace_urls, replace_emails, replace_numbers, replace_currency_symbols, remove_english_alphabets
loaded_model = pickle.load(open("traind_LR_classifier.pkle", "rb"))
from typing  import FrozenSet
#Urdu language stop words list
STOP_WORDS: FrozenSet[str] = frozenset("""
 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
""".split())

def remove_stopwords(text: str):
    return " ".join(word for word in text.split() if word not in STOP_WORDS)

from urduhack.models.lemmatizer import lemmatizer

def lemmatize_str(input_str):
    lemme_str = ""
    temp = lemmatizer.lemma_lookup(input_str)
    for t in temp:
        lemme_str += t[0] + " "

    return lemme_str.strip()  # Remove trailing space

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
    df_new['lemmatized_text'] = df_new['review'].apply(lemmatize_str)

    # Apply TF-IDF Vectorization
    max_feature_num = 50000
    vectorizer = TfidfVectorizer(max_features=max_feature_num)
    
    vectorizer.fit(df_new['lemmatized_text'])
    new_test_vecs = vectorizer.transform(df_new['lemmatized_text'])
  

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
