import nltk
import os
import re
import math
import operator
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from IPython.display import display, HTML

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
Stopwords = set(stopwords.words('indonesian'))
wordlemmatizer = WordNetLemmatizer()

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex, '', text)
    return text

def custom_stopwords_removal(words):
    additional_stopwords = [
        "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 
        'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 
        'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 
        'aja', 'n', 't','tks', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', 
        '&amp', 'yah','wkwk',"assalamu'alaikum",'wr','wb','bapak','ibu', 'wabillahi',
        'taufik','wal','hidayah',"wassalamu'alaikum", 'yth', 'puji','syukur',
        'hadirat','allah','swt','karunia', 'selamat','pagi','siang','approval','close',
        'forum','atau', 'halaman','yang','dan','terima','kasih','mohon','terimakasih',
        'bapakibu','bpkibu','dengan','maaf','bapak','ibu','atau'
    ]
    nltk_stopwords = set(stopwords.words('indonesian'))
    list_stopwords = set(nltk_stopwords).union(additional_stopwords)
    return list_stopwords

def stopwords_removal(words, list_stopwords):
    clean_words = [word for word in words if word.lower() not in list_stopwords]
    return clean_words

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = list(set(words))
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = [word for word, tag in pos_tag if tag.startswith("NN") or tag.startswith("VB")]
    return pos_tagged_noun_verb

def preprocess_text(text):
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    return ' '.join(tokenized_words)

def generate_tfidf_matrix(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

def hybrid_tfidf_score(tf_matrix, sentence_index, word_freq, sentences, threshold_minimal):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    tf_score = tf_score_sentence(X.toarray(), sentence_index, vectorizer)
    
    idf_score = calculate_idf(sentences)
    idf_scores = calculate_sentence_idf(sentences, idf_score)
    
    hybrid_tfidf_score = tf_score * idf_scores[sentence_index]['Sentence_IDF'] / threshold_minimal
    return hybrid_tfidf_score


def tf_score_word(tf_matrix, sentence_index, target_word, vectorizer):
    word_index = vectorizer.vocabulary_[target_word]
    tf_score = tf_matrix[sentence_index, word_index] / tf_matrix[sentence_index].sum()
    return tf_score

def tf_score_sentence(tf_matrix, sentence_index, vectorizer):
    tf_scores = [tf_score_word(tf_matrix, sentence_index, word, vectorizer) for word in vectorizer.get_feature_names_out()]
    return sum(tf_scores) / len(tf_scores) if len(tf_scores) > 0 else 0

def calculate_sentence_idf(preprocessed_sentences, idf_scores):
    sentence_idf_scores = []

    for i, document in enumerate(preprocessed_sentences, start=1):
        words = document.split()
        sentence_idf = sum(idf_scores[word] for word in set(words))
        sentence_idf_scores.append({'Sentence': f'Sentence_{i}', 'Sentence_IDF': sentence_idf})

    return sentence_idf_scores

def calculate_idf(preprocessed_sentences):
    N = len(preprocessed_sentences)
    idf_scores = defaultdict(float)
    total_words_per_document = {}

    for i, document in enumerate(preprocessed_sentences, start=1):
        words = document.split()
        total_words_per_document[i] = len(words)

        unique_words = set(words)
        for word in unique_words:
            idf_scores[word] += 1

    for word, df in idf_scores.items():
        idf_scores[word] = math.log(N / (df + 1)) + 1 

    return idf_scores

def calculate_average_words(sentences):
    total_words = sum(len(sentence.split()) for sentence in sentences)
    average_words = total_words / len(sentences)
    return average_words

def calculate_threshold(sentences):
    average_words = calculate_average_words(sentences)
    threshold_minimal = int(average_words)
    return threshold_minimal

def run_hybrid_tf_idf(text):
    tokenized_sentence = sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in tokenized_sentence]
    
    tfidf_matrix = generate_tfidf_matrix(preprocessed_sentences)
    
    all_words = [word for sentence in preprocessed_sentences for word in word_tokenize(sentence)]
    word_freq = freq(all_words)
    
    input_user = 30
    no_of_sentences = int((input_user * len(tokenized_sentence)) / 100)
    
    threshold_minimal = calculate_threshold(preprocessed_sentences)
    
    vectorizer = CountVectorizer()
    tfidf_matrix = generate_tfidf_matrix(preprocessed_sentences)
    
    c = 1
    sentence_with_importance = {}
    for i in range(len(tokenized_sentence)):
    	hybrid_tfidf = hybrid_tfidf_score(tfidf_matrix, i, word_freq, preprocessed_sentences, threshold_minimal)
    	sentence_with_importance[c] = hybrid_tfidf
    	c = c + 1
    
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    for word_prob in sentence_with_importance:
    	if cnt < no_of_sentences:
        	sentence_no.append(word_prob[0])
        	cnt = cnt + 1
    	else:
        	break
    sentence_no.sort()
    cnt = 1
    for sentence in tokenized_sentence:
    	if cnt in sentence_no:
    		summary.append(sentence)
    	cnt = cnt + 1
    summary = " ".join(summary)
    return summary
