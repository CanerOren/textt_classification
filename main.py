'''import os
import string
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from jpype import (JClass, JString, getDefaultJVMPath,shutdownJVM,startJVM,java)
import pandas as pd
import re
import jpype
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score,classification_report
paths = [r"D:\Yazılım\textt_classification\1",r"D:\Yazılım\textt_classification\2",r"D:\Yazılım\textt_classification\3"]
ZEMBEREK_PATH = 'zemberek-full_old.jar'
DATA_PATH="data"
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer:JClass=JClass("zemberek.normalization.TurkishSentenceNormalizer")
Paths: JClass = JClass("java.nio.file.Paths")
morphology = TurkishMorphology.createWithDefaults()
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
    Paths.get(str(os.path.join(DATA_PATH, "lm", "lm.2gram.slm"))),
)


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def tokenizasyon(text):
    return word_tokenize(text)

def normalizasyon(text):
    normalized_words=[]
    for text in tokenizasyon(text):
        normalized_word=str(normalizer.normalize(JString(text)))
        normalized_words.append(normalized_word)
    text=' '.join(normalized_words)
    return text



def lemmatizer(text):
    lemma_words = []


    for word in tokenizasyon(text):
        lemma_word = str(morphology.analyzeAndDisambiguate(str(word)).bestAnalysis()[0].getLemmas()[0])
        lemma_words.append(lemma_word)
    text = ' '.join(lemma_words)
    return text

def convert_lowercase(text):
    return text.lower()
def remove_punctuation(text):
    return ''.join(d for d in text if d not in string.punctuation)
def remove_stopwords(text):
    stopwords = []
    with open(r'D:\Yazılım\textt_classification\stop-words_turkish_1_tr.txt', 'r',encoding='utf-8') as f:
        for word in f:
            word = word.split('\n')
            stopwords.append(word[0])
    clean_text = ' '.join(s for  s in text.split() if s not in stopwords)
    return clean_text
def remove_numbers(text):
    text = re.sub(r'\d', '', text)
    return text

def remove_extra_space(text):
    ornek_text_strip = re.sub(' +', ' ', text)
    return ornek_text_strip.strip()

def remove_less_than_2(text):
    text = ' '.join([w for w in text.split() if len(w)>2])
    return text

df=pd.DataFrame(columns=['Cümle','Kategori'])
list=[]
list2=[]
for path in paths:
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            list.append(str(read_text_file(file_path)))
            list2.append(file)
df["Cümle"]=list

df["Kategori"].iloc[0:756]='Olumlu'
df["Kategori"].iloc[756:2043]='Olumsuz'
df["Kategori"].iloc[2043:3001]='Nötr'


df["Cümle"]=df["Cümle"].apply(normalizasyon)
df["Cümle"]=df["Cümle"].apply(lemmatizer)
df["Cümle"]=df["Cümle"].apply(convert_lowercase)
df["Cümle"]=df["Cümle"].apply(remove_punctuation)
df["Cümle"] = df['Cümle'].apply(remove_stopwords)
df['Cümle'] = df['Cümle'].apply(remove_extra_space)
df['Cümle'] = df['Cümle'].apply(remove_numbers)
df['Cümle'] = df['Cümle'].apply(remove_less_than_2)

X=df["Cümle"]
y=df["Kategori"]

tfidf=TfidfVectorizer()
X_tfidf=tfidf.fit_transform(X)

chi2_features = SelectKBest(chi2, k = 1000)
X_tfidf = chi2_features.fit_transform(X_tfidf, y)


clf=MultinomialNB()
cv=StratifiedKFold(n_splits=10,shuffle=True)

eval_metrics=[]

for i,(train_idx,test_idx) in enumerate(cv.split(X_tfidf,y)):
    X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)
    eval_metrics.append(classification_report(y_test,y_pred))

with open(r'sonuc.txt','w') as f:
    f.write('Evaluation Metrics:\n')
    for i, report in enumerate(eval_metrics):
        f.write('Folds:\t')
        f.write(str(i+1))
        f.write('\n')
        f.write(str(report))
        f.write('\n')
feature=[]
list2.pop(-1)
kategori_list=y.values.tolist()
kategori_list.pop(-1)
for i in range(0,3277):
    if chi2_features._get_support_mask()[i]==True:
        feature.append(tfidf.get_feature_names_out()[i])
tf_idf = pd.DataFrame(X_tfidf.todense()).iloc[:3000]
tf_idf.columns = feature
tfidf_matrix = tf_idf.T
tfidf_matrix.columns = list2
tfidf_matrix=tfidf_matrix.T
tfidf_matrix["Kategori"]=kategori_list


tfidf_matrix.to_csv('tfidf_matrix.csv')'''
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from jpype import (JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java)
import pandas as pd
import re
import jpype
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report

paths = [r"D:\Yazılım\textt_classification\1", r"D:\Yazılım\textt_classification\2", r"D:\Yazılım\textt_classification\3"]
ZEMBEREK_PATH = 'zemberek-full_old.jar'
DATA_PATH = "data"
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer: JClass = JClass("zemberek.normalization.TurkishSentenceNormalizer")
Paths: JClass = JClass("java.nio.file.Paths")
morphology = TurkishMorphology.createWithDefaults()
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
    Paths.get(str(os.path.join(DATA_PATH, "lm", "lm.2gram.slm"))),
)

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def tokenizasyon(text):
    return word_tokenize(text)

def normalizasyon(text):
    normalized_words = []
    for text in tokenizasyon(text):
        normalized_word = str(normalizer.normalize(JString(text)))
        normalized_words.append(normalized_word)
    text = ' '.join(normalized_words)
    return text

def lemmatizer(text):
    lemma_words = []
    for word in tokenizasyon(text):
        lemma_word = str(morphology.analyzeAndDisambiguate(str(word)).bestAnalysis()[0].getLemmas()[0])
        lemma_words.append(lemma_word)
    text = ' '.join(lemma_words)
    return text

def convert_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return ''.join(d for d in text if d not in string.punctuation)

def remove_stopwords(text):
    stopwords = []
    with open(r'D:\Yazılım\textt_classification\stop-words_turkish_1_tr.txt', 'r', encoding='utf-8') as f:
        for word in f:
            word = word.split('\n')
            stopwords.append(word[0])
    clean_text = ' '.join(s for s in text.split() if s not in stopwords)
    return clean_text

def remove_numbers(text):
    text = re.sub(r'\d', '', text)
    return text

def remove_extra_space(text):
    ornek_text_strip = re.sub(' +', ' ', text)
    return ornek_text_strip.strip()

def remove_less_than_2(text):
    text = ' '.join([w for w in text.split() if len(w) > 2])
    return text

df = pd.DataFrame(columns=['Cümle', 'Kategori'])
list = []
list2 = []
for path in paths:
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            list.append(str(read_text_file(file_path)))
            list2.append(file)
df["Cümle"] = list

df["Kategori"].iloc[0:756] = 'Olumlu'
df["Kategori"].iloc[756:2043] = 'Olumsuz'
df["Kategori"].iloc[2043:3001] = 'Nötr'

df["Cümle"] = df["Cümle"].apply(normalizasyon)
df["Cümle"] = df["Cümle"].apply(lemmatizer)
df["Cümle"] = df["Cümle"].apply(convert_lowercase)
df["Cümle"] = df["Cümle"].apply(remove_punctuation)
df["Cümle"] = df['Cümle'].apply(remove_stopwords)
df['Cümle'] = df['Cümle'].apply(remove_extra_space)
df['Cümle'] = df['Cümle'].apply(remove_numbers)
df['Cümle'] = df['Cümle'].apply(remove_less_than_2)

X = df["Cümle"]
y = df["Kategori"]

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

chi2_features = SelectKBest(chi2, k=1000)
X_tfidf = chi2_features.fit_transform(X_tfidf, y)

# Seçilen özellik maskesini kontrol edin ve isimlerini alın
selected_features_mask = chi2_features.get_support()
selected_feature_names = tfidf.get_feature_names_out()[selected_features_mask]

# Seçilen özellik sayısını ve isimlerini yazdır
print("Seçilen Özellik Sayısı:", selected_features_mask.sum())
print("Seçilen Özellik İsimleri:", selected_feature_names)

clf = MultinomialNB()
cv = StratifiedKFold(n_splits=10, shuffle=True)

eval_metrics = []

for i, (train_idx, test_idx) in enumerate(cv.split(X_tfidf, y)):
    X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    eval_metrics.append(classification_report(y_test, y_pred))

with open(r'sonuc.txt', 'w') as f:
    f.write('Evaluation Metrics:\n')
    for i, report in enumerate(eval_metrics):
        f.write('Folds:\t')
        f.write(str(i + 1))
        f.write('\n')
        f.write(str(report))
        f.write('\n')

feature = []
list2.pop(-1)
kategori_list = y.values.tolist()
kategori_list.pop(-1)

for i in range(len(selected_features_mask)):
    if selected_features_mask[i]:
        feature.append(tfidf.get_feature_names_out()[i])

tf_idf = pd.DataFrame(X_tfidf.todense()).iloc[:3000]
tf_idf.columns = feature
tfidf_matrix = tf_idf.T
tfidf_matrix.columns = list2
tfidf_matrix = tfidf_matrix.T
tfidf_matrix["Kategori"] = kategori_list

tfidf_matrix.to_csv('tfidf_matrix.csv')
