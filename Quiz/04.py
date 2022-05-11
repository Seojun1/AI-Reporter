import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)

sequences = []
max_len = 0

for i in range(len(titles)):
    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
    sequences.append(sequence)
    max_len = max(max_len, len(sequence))

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
categorical_data = tf.keras.utils.to_categorical(sequence, num_classes=10000)
print(pad_sequences)
print(categorical_data)