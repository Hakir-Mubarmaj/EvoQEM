#!/usr/bin/env python3
"""
train_circuit_transformer_fixed.py
- Fixed IndexError by adding bound checks.
- Added 'mask_zero=True' to improve accuracy (ignores padding).
- Handles TensorFlow/Keras imports robustly.
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf

# Suppress TensorFlow INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from qiskit import qpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.utils import to_categorical

# ---------------- Configuration ----------------
QPY_FILE = 'test.qpy'        # Ensure this matches your file name
CSV_FILE = 'test_fidelity_analysis_true_only.csv' 
MODEL_SAVE_PATH = 'circuit_translator_model.h5'

BATCH_SIZE = 32
EPOCHS = 20
LATENT_DIM = 256
MAX_SEQ_LEN = 300  # Increased length to handle larger circuits

# ---------------- Helper Functions ----------------
def circuit_to_string(qc):
    """
    Converts a QuantumCircuit into a simplified string sequence of gates.
    """
    ops = []
    for instruction in qc.data:
        op_name = instruction.operation.name
        qubits = [str(qc.find_bit(q).index) for q in instruction.qubits]
        token = f"{op_name}_{'_'.join(qubits)}"
        ops.append(token)
    return ' '.join(ops)

def main():
    # 1. Load Data
    print(f"Loading circuits from {QPY_FILE}...")
    if not os.path.exists(QPY_FILE):
        print(f"Error: {QPY_FILE} not found.")
        return

    with open(QPY_FILE, 'rb') as fd:
        all_circuits = qpy.load(fd)
    
    print(f"Loading mapping from {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # 2. Prepare Pairs
    input_texts = []
    target_texts = []
    
    print("Preprocessing circuit sequences...")
    for _, row in df.iterrows():
        main_idx = int(row['circuit_id'])
        target_idx = int(row['best_match_id'])
        
        qc_in = all_circuits[main_idx]
        qc_out = all_circuits[target_idx]
        
        in_str = circuit_to_string(qc_in)
        out_str = 'start_ ' + circuit_to_string(qc_out) + ' _end'
        
        input_texts.append(in_str)
        target_texts.append(out_str)
        
    print(f"Total training pairs: {len(input_texts)}")

    # 3. Tokenization
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(input_texts + target_texts)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    encoder_sequences = tokenizer.texts_to_sequences(input_texts)
    decoder_sequences = tokenizer.texts_to_sequences(target_texts)
    
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_SEQ_LEN, padding='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_SEQ_LEN, padding='post')
    
    decoder_target_data = np.zeros((len(input_texts), MAX_SEQ_LEN, vocab_size), dtype='float32')
    
    # FIX 1: Safety check in loop
    for i, seq in enumerate(decoder_sequences):
        for t, token_id in enumerate(seq):
            if t > 0 and t < MAX_SEQ_LEN:
                decoder_target_data[i, t - 1, token_id] = 1.0

    # 4. Build Seq2Seq Model
    
    # Encoder
    encoder_inputs = Input(shape=(MAX_SEQ_LEN,))
    # FIX 2: mask_zero=True ignores 0 padding, improving accuracy
    enc_emb = Embedding(vocab_size, LATENT_DIM, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(MAX_SEQ_LEN,))
    # FIX 2: mask_zero=True here as well
    dec_emb_layer = Embedding(vocab_size, LATENT_DIM, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Compile
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()

    # 5. Train
    print("Starting training...")
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2
    )
    
    model.save(MODEL_SAVE_PATH) # H5 format
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # ---------------- Inference Setup ----------------
    
    # Re-build Encoder for Inference
    encoder_model = Model(encoder_inputs, encoder_states)

    # Re-build Decoder for Inference
    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2
    )

    # FIX 3: Robust Decoding Loop
    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq, verbose=0)

        target_seq = np.zeros((1, MAX_SEQ_LEN))
        target_seq[0, 0] = tokenizer.word_index.get('start_', 1)

        stop_condition = False
        decoded_sentence = []
        current_idx = 0
        
        while not stop_condition:
            # Check bounds immediately
            if current_idx >= MAX_SEQ_LEN - 1:
                break

            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

            # Get the token for the current step
            sampled_token_index = np.argmax(output_tokens[0, current_idx, :])
            sampled_word = tokenizer.index_word.get(sampled_token_index, '')

            if sampled_word == '_end' or not sampled_word:
                stop_condition = True
            else:
                decoded_sentence.append(sampled_word)

            # Update target sequence for next step
            if current_idx + 1 < MAX_SEQ_LEN:
                target_seq[0, current_idx + 1] = sampled_token_index
            
            states_value = [h, c]
            current_idx += 1

        return ' '.join(decoded_sentence)

    # Test
    print("\n--- Testing Model Predictions ---")
    for i in range(min(3, len(encoder_input_data))):
        input_seq = encoder_input_data[i:i+1]
        decoded = decode_sequence(input_seq)
        print(f"Input:  {input_texts[i][:100]}...") # Print first 100 chars
        print(f"Actual: {target_texts[i][:100]}...")
        print(f"Pred:   {decoded}")
        print("-" * 30)

if __name__ == "__main__":
    main()