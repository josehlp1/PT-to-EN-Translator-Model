import pandas as pd
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import pickle

# Baixar e descompactar o dataset
url_sentences = 'https://downloads.tatoeba.org/exports/sentences.csv'
urllib.request.urlretrieve(url_sentences, 'sentences.csv')

url_links = 'https://downloads.tatoeba.org/exports/links.csv'
urllib.request.urlretrieve(url_links, 'links.csv')

# Carregar os datasets
sentences = pd.read_csv('sentences.csv', delimiter='\t', header=None, names=['id', 'lang', 'text'])
links = pd.read_csv('links.csv', delimiter='\t', header=None, names=['id1', 'id2'])

# Filtrar frases em inglês e português
eng_sentences = sentences[sentences['lang'] == 'eng']
por_sentences = sentences[sentences['lang'] == 'por']

# Mesclar os datasets de links com as frases
eng_links = links.merge(eng_sentences, left_on='id1', right_on='id')
por_links = eng_links.merge(por_sentences, left_on='id2', right_on='id')

# Selecionar e renomear as colunas
dataset = por_links[['text_x', 'text_y']]
dataset.columns = ['english', 'portuguese']

# Aumentar o tamanho do dataset para 50k linhas
dataset = dataset.sample(n=60000, random_state=42)

# Adicionar tokens especiais
start_token = '<start>'
end_token = '<end>'
dataset['portuguese'] = dataset['portuguese'].apply(lambda x: f"{start_token} {x} {end_token}")

# Tokenização
tokenizer_eng = Tokenizer(filters='')
tokenizer_por = Tokenizer(filters='')

# Garantir que os tokens especiais sejam adicionados ao vocabulário
tokenizer_eng.fit_on_texts([start_token, end_token])
tokenizer_por.fit_on_texts([start_token, end_token])

# Fit do tokenizador nos textos
tokenizer_eng.fit_on_texts(dataset['english'])
tokenizer_por.fit_on_texts(dataset['portuguese'])

# Salvar os tokenizadores
with open('tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(tokenizer_eng, f)

with open('tokenizer_por.pkl', 'wb') as f:
    pickle.dump(tokenizer_por, f)

# Sequenciar e padronizar
sequences_eng = tokenizer_eng.texts_to_sequences(dataset['english'])
sequences_por = tokenizer_por.texts_to_sequences(dataset['portuguese'])

max_len = 50

sequences_eng = pad_sequences(sequences_eng, maxlen=max_len, padding='post', truncating='post')
sequences_por = pad_sequences(sequences_por, maxlen=max_len, padding='post', truncating='post')

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(sequences_eng, sequences_por, test_size=0.2, random_state=42)

latent_dim = 256
num_encoder_tokens = len(tokenizer_eng.word_index) + 1
num_decoder_tokens = len(tokenizer_por.word_index) + 1

encoder_inputs = Input(shape=(max_len,))
x = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim, input_length=max_len)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_len,))
x = Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim, input_length=max_len)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

decoder_input_data = np.zeros_like(y_train)
decoder_input_data[:, 1:] = y_train[:, :-1]
decoder_input_data[:, 0] = tokenizer_por.word_index[start_token]

decoder_target_data = np.expand_dims(y_train, -1)
model.fit([X_train, decoder_input_data], decoder_target_data, epochs=20, batch_size=64, validation_split=0.2)
model.save('nmt_model.h5')

decoder_input_data_test = np.zeros_like(y_test)
decoder_input_data_test[:, 1:] = y_test[:, :-1]
decoder_input_data_test[:, 0] = tokenizer_por.word_index[start_token]

decoder_target_data_test = np.expand_dims(y_test, -1)

loss, accuracy = model.evaluate([X_test, decoder_input_data_test], decoder_target_data_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
