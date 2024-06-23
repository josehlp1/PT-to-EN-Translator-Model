from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Defina latent_dim (o mesmo valor usado no treino)
latent_dim = 256
max_len = 50
start_token = '<start>'
end_token = '<end>'

# Carregar tokenizadores salvos
with open('tokenizer_eng.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)

with open('tokenizer_por.pkl', 'rb') as f:
    tokenizer_por = pickle.load(f)

# Função para decodificar sequência
def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer_por, max_len, start_token, end_token):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_por.word_index[start_token]

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer_por.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_token

        if (sampled_token == end_token or len(decoded_sentence.split()) >= max_len):
            stop_condition = True

        # Atualizar a sequência de alvo (de comprimento 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()

# Função para preparar a entrada
def prepare_input(text, tokenizer_eng, max_len):
    sequence = tokenizer_eng.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_len, padding='post')

# Carregar o modelo salvo
model = load_model('nmt_model.h5')

# Reconstruir os modelos de encoder e decoder
encoder_inputs = model.input[0]
encoder_embedding_layer = model.get_layer(name='embedding')
encoder_lstm_layer = model.get_layer(name='lstm')

encoder_embedding_output = encoder_embedding_layer(encoder_inputs)
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm_layer(encoder_embedding_output)
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_embedding_layer = model.get_layer(name='embedding_1')
decoder_lstm_layer = model.get_layer(name='lstm_1')
decoder_dense_layer = model.get_layer(name='dense')

decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,), name='input_5')
decoder_inputs_single_x = decoder_embedding_layer(decoder_inputs_single)
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
    decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense_layer(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Testar o modelo com algumas frases
test_sentences = [
    "How are you?",
    "I love programming.",
    "This is a test sentence.",
    "The weather is nice today."
]

for sentence in test_sentences:
    input_seq = prepare_input(sentence, tokenizer_eng, max_len)
    translated_sentence = decode_sequence(input_seq, encoder_model, decoder_model, tokenizer_por, max_len, start_token, end_token)
    print(f"English: {sentence}")
    print(f"Portuguese: {translated_sentence}")
