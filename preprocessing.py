#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[ ]:


df = pd.read_csv('train.csv')
df.head()


# In[ ]:


#import stopword
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# In[ ]:


def pre_process(text):
    # Case Folding: Lowercase
    # Merubah format teks menjadi format huruf kecil semua (lowercase).
    text = text.lower()

    # Case Folding: Removing Number
    # Menghapus karakter angka.
    text = re.sub(r"\d+", "", text)

    # Case Folding: Removing Punctuation
    # Menghapus karakter tanda baca.
    text = text.translate(str.maketrans("","",string.punctuation))

    #Case Folding: Removing Whitespace
    #Menghapus karakter kosong.
    text = text.strip()

    
    #Separating Sentences with Split () Method
    #Fungsi split() memisahkan string ke dalam list dengan spasi sebagai pemisah jika tidak ditentukan pemisahnya.
    pisah = text.split()

    #Filtering using sastrawi
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text =  stopword.remove(text)

    return text

df['ulasan'] = df['ulasan'].apply(lambda x:pre_process(x))
df.head()


# In[ ]:


df['label'].value_counts()


# In[ ]:


df['label'].value_counts().sort_index().plot.bar()


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, Flatten
from keras.layers import Bidirectional
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
train_df, test_df = train_test_split(df, test_size = 0.25, random_state = 42)
print("Training data size : ", train_df.shape)
print("Test data size : ", test_df.shape)


# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
train_df['ulasan']=train_df['ulasan'].astype(str)
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(train_df['ulasan'].values)

X = tokenizer.texts_to_sequences(train_df['ulasan'].values)
X = pad_sequences(X) # padding our text vector so they all have the same length
X[:5]


# In[ ]:


top_words = 500
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(train_df['ulasan'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['ulasan'])

max_review_length = 200
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
y_train = train_df['label']


# In[ ]:


embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(64))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train,y_train, epochs=20, batch_size=64, validation_split=0.2)


# In[ ]:


model.save('sentiment_analysis.h5')


# In[ ]:


list_tokenized_test = tokenizer.texts_to_sequences(test_df['ulasan'])
X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)
y_test = test_df['label']
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


confusion_matrix(y_test,y_pred, labels=[1,0])


# In[ ]:


TP = 216
FN = 0
FP = 35
TN = 0
print(classification_report(y_test,y_pred))


# In[ ]:


#menghitung nilai presisi recall, f-1 score model kita ddalam memprediksi data yg positif
precision = TP/(TP+FP)
print(precision)


# In[ ]:


recall = TP/(TP+FN)
print(recall)


# In[ ]:


f1score = 2*precision*recall/(precision+recall)
print(f1score)


# In[ ]:


#menghitung nilai presisi recall, f-1 score model kita ddalam memprediksi data yg negatif
precision = TN/(TN+FN)
print(precision)


# In[ ]:


recall = TN/(TN+FP)
print(recall)


# In[ ]:


f1score = 2*precision*recall/(precision+recall)
print(f1score)

