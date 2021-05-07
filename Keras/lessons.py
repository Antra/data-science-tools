# Lesson2

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import zipfile
import requests


data_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip'


def get_data(url, fn):
    response = requests.get(url)
    f = open(fn, 'wb')
    f.write(response.content)
    f.close()


def extract_data(fn):
    with zipfile.ZipFile(fn, 'r') as zf:
        zf.extractall()


data_set_file_name = 'dataset.zip'
get_data(data_set_url, data_set_file_name)
extract_data(data_set_file_name)


def rename_data_folder(fn, new_name):
    if os.path.exists(fn) and not os.path.exists(new_name):
        os.rename(fn, new_name)


data_set_folder = 'sentiment labelled sentences'

rename_data_folder(data_set_folder, 'data')


filepath_dict = {
    'yelp': 'data/yelp_labelled.txt',
    'amazon': 'data/amazon_cells_labelled.txt',
    'imdb': 'data/imdb_labelled.txt'
}

df_list = []

for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)

print(df.head())

# Lesson 3
# Terminology
# - Corpus = collection of texts being examined
# - Vocabulary = collection of unique words being used in the corpus
# -- Unique words, because each word is assigned a unique ID for referencing during training
# - Feature vector = Representation per sentence (row) using the unique word IDs (aka a numerical representation of each sentence)

# This kind of model is called a Bag-of-Words (BoW) model

sentences = ['John likes ice cream', 'John hates chocolate']

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
print(vectorizer.vocabulary_)

print(vectorizer.transform(sentences).toarray())

# Lesson 4
df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
labels = df_yelp['label'].values


sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.25, random_state=1000)


vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

X_train


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print(score)


for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    labels = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    print(f'{source}: {score:.4f}')


# Lesson 6
df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
labels = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

X_train.shape


model = models.Sequential()
model.add(layers.Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)


_, train_accuracy = model.evaluate(X_train, y_train)
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Training accuracy: {train_accuracy:.4f}')
print(f'Testing accuracy: {test_accuracy:.4f}')


plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


plot_history(history)


type(history.history)

len(df)


# Lesson 7

cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']


encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)

city_labels


one_hot_encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
one_hot_encoder.fit_transform(city_labels)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

print(sentences_train[2])
print(X_train[2])


X_train = pad_sequences(X_train, padding='post', maxlen=100)
X_test = pad_sequences(X_test, padding='post', maxlen=100)

print(sentences_train[2])
print(X_train[2])


model = models.Sequential()
model.add(
    layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=50,
        input_length=100
    )
)
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_test, y_test), batch_size=10)


_, train_accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy:.4f}')
plot_history(history)


model = models.Sequential()
model.add(
    layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=50,
        input_length=100
    )
)
model.add(layers.GlobalMaxPool1D())
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    batch_size=10
)

_, train_accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy:.4f}')
plot_history(history)


# Lesson 8
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'

GLOVE_FILE_NAME = 'glove.zip'

get_data(GLOVE_URL, GLOVE_FILE_NAME)
extract_data(GLOVE_FILE_NAME)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                    :embedding_dim]

    return embedding_matrix


embedding_matrix = create_embedding_matrix(
    'glove.6B.50d.txt', tokenizer.word_index, 50)

embedding_matrix.shape

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / (len(tokenizer.word_index) + 1)


model = models.Sequential()
model.add(layers.Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=50,
    input_length=100,
    weights=[embedding_matrix],
    trainable=False))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=10
)

_, train_accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy:.4f}')
plot_history(history)


model = models.Sequential()
model.add(layers.Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=50,
    input_length=100,
    weights=[embedding_matrix],
    trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=10
)

_, train_accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy:.4f}')
plot_history(history)


# Lesson 9
model = models.Sequential()
model.add(layers.Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=100,
    input_length=100))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=10
)

_, train_accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')
_, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy:.4f}')
plot_history(history)


def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


param_grid = dict(
    num_filters=[32, 64, 128],
    kernel_size=[3, 5, 7],
    vocab_size=[5000],
    embedding_dim=[50],
    maxlen=[100]
)


# Main settings
epochs = 20
embedding_dim = 50
maxlen = 100
output_file = 'data/output.txt'

# Run grid search for each source (yelp, amazon, imdb)
for source, frame in df.groupby('source'):
    print('Running grid search for data set :', source)
    sentences = df['sentence'].values
    y = df['label'].values

    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)

    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)

    # Save and evaluate results
    prompt = input(f'finished {source}; write to file and proceed? [y/n]')
    if prompt.lower() not in {'y', 'true', 'yes'}:
        break
    with open(output_file, 'a') as f:
        s = ('Running {} data set\nBest Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            source,
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
        print(output_string)
        f.write(output_string)
