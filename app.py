import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse

nltk.download('stopwords')
ps = PorterStemmer()

# Your preprocessing function stays the same
def transform_text(text):
    token = text.lower().split()
    y = [word for word in token if word.isalnum()]
    y = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    y = [ps.stem(word) for word in y]
    return " ".join(y)

# Add DenseTransformer class here BEFORE loading model
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        if issparse(x):
            return x.toarray()
        else:
            return x

# Load vectorizer and your saved ensemble model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a non-empty message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
else:
    st.write("Enter a message and click Predict.")
