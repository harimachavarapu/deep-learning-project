#importing requried laibraies
import pandas as pd
import re 
from wordcloud import WordCloud,STOPWORDS
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from PIL import Image
###########################
#fornt end desinging
# First some code.
primaryColor = '#9BE09B'
backgroundColor = '#00FFE8'
secondaryBackgroundColor = '#7575E2'
textColor = '#F3B48D'
font = "monospace"
image = Image.open('C:/Users/hudso/OneDrive/Pictures/wow/coww.jpg')
st.image(image, caption='HARI KRISHNA SRI SAI PRASAD MACHAVARAPU')#WE CAN INSERT OUR IMAGE
st.title("FEATURES OF DIFFERENT TYPES APPLICATIONS IN PLAYSTORE")
st.subheader("MODEL IS BASED ON APP'S DESCRIPTION")
st.caption('HARI KRISHNA SRI SAI PRASSAD')
st.snow()#ADDS SNOW EFFECTS
st.balloons()#ADD BALLONS
############################
GENRE = st.selectbox('CHOOSE GENRE:',('Productivity','Education','Communication','Tools','Maps & Navigation',
                                      'News & Magazines','Business','Entertainment','Travel & Local','Finance',
                                      'Social','Shopping','Comics','Video Players & Editors','Lifestyle','Food & Drink'
                                     ' Art & Design','Health & Fitness','Books & Reference','Puzzle')) 
NGRAM = st.selectbox('SELECT NGRAM',('1', '2', '3')) 

df=pd.read_csv("C:/Users/hudso/harikrishnasdsm.csv") #load the dataset
data = df.describe()
genrechoice = df.genre.unique()#we can only choose defult genere 1
description = df["description"].loc[df["genre"] == GENRE]#coloumn can be changed
ngram = df["description"].loc[df["genre"] == NGRAM]#divides description into grams

if NGRAM == '1':
    ip_rev_string = " ".join(description)
    ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
    ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
    
    ip_reviews_words = ip_rev_string.split(" ")
    ip_reviews_words = ip_reviews_words[1:]
    
    from sklearn.feature_extraction.text import CountVectorizer
    with open("C:/Users/hudso/OneDrive/Desktop/assingmentd/stopwords_en.txt", "r") as sw:
     stop_words = sw.read()  
    ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]
    # Joinining all the reviews into single paragraph 
    ip_rev_string = " ".join(ip_reviews_words)
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    bag_of_words = vectorizer.fit_transform(ip_reviews_words)
    # Using count vectoriser to view the frequency 
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    st.table (words_freq[:100])
    #displays output in table format

elif NGRAM =='2':
    nltk.download('punkt')
    WNL = nltk.WordNetLemmatizer()
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    
    ip_rev_string = " ".join(description)
    # Lowercase and tokenize
    text = ip_rev_string.lower()

    # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")

    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)

    # Remove extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

    # Create a set of stopwords
    stopwords_wc = set(STOPWORDS)
    customised_words = [] # If you want to remove any particular word form text which does not contribute much in meaning


    new_stopwords = stopwords_wc.union(customised_words)

    # Remove stop words
    text_content = [word for word in text_content if word not in new_stopwords]

    # Take only non-empty entries
    text_content = [s for s in text_content if len(s) != 0]

    # Best to get the lemmas of each word to reduce the number of similar words
    text_content = [WNL.lemmatize(t) for t in text_content]

    # nltk_tokens = nltk.word_tokenize(text)  
    bigrams_list = list(nltk.bigrams(text_content))
    
    dictionary2 = [' '.join(tup) for tup in bigrams_list]

    # Using count vectoriser to view the frequency of bigrams

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
   

    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    st.table (words_freq[:100])
    #displays output in table format
else:
    NGRAM =='3'
    ip_rev_string = " ".join(description)
     # Lowercase and tokenize 
    text = ip_rev_string.lower()

     # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")

    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)
     # Remove extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

     # Create a set of stopwords
    stopwords_wc = set(STOPWORDS)
    customised_words = [] # If you want to remove any particular word form text which does not contribute much in meaning

    new_stopwords = stopwords_wc.union(customised_words)

     # Remove stop words
    text_content = [word for word in text_content if word not in new_stopwords]

     # Take only non-empty entries
    text_content = [s for s in text_content if len(s) != 0]
    WNL = nltk.WordNetLemmatizer()
     # Best to get the lemmas of each word to reduce the number of similar words
    text_content = [WNL.lemmatize(t) for t in text_content]

     # nltk_tokens = nltk.word_tokenize(text)  
    trigrams_list = list(nltk.trigrams(text_content))
    
    dictionary3 = [' '.join(tup) for tup in trigrams_list]

     # Using count vectoriser to view the frequency of bigrams
    
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    bag_of_words = vectorizer.fit_transform(dictionary3)


    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    st.table(words_freq[:100])
    #display output in table format
