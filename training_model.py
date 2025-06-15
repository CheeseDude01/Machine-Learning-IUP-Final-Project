# train_with_full_metrics.py

import pandas as pd
import numpy as np
import re # for text cleansing
import json # for config
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow informational messages

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score # <-- Import new metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras import backend as K

# tensorflow and keras for building the model and build neural network
# sklearn for metrics

# --- Constants ---
VOCAB_SIZE = 10000
MAX_LEN = 0

# --- Helper Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', '', text)
    return text

# --- Model Building Functions ---


'''
    we use a sequential model, which is a stack of layer that includes embedding, dropouy (prevents memorization), Conv1D (to find pattern in CNN), 
    Bidirectional LSTM (understand context), and Dense (final decision)
    we stack the layers to build a deep learning architecture that learn pattern from the text 
    the result is the keras model from each CNN and BiLSTM ready for training


    embedding layer translates numerical ID for depression and non depression to a concept the model would understand
    dropout layer is for force the model to learn the underlying concepts robustly. if repeatedly trained on the same data, the model would become really good
    that leads to overfitting
    globalmaxpooling1d: points to the single most critical phrase.
    dense layer are standard fully connected neural network layers that take high level feature and learn how to weight them 
    the sigmoid activation function decides any input value to be either 1 or 0

'''

def build_cnn_model(embedding_dim=128, dropout_rate=0.3):
    K.clear_session()
    model = Sequential([
        Embedding(VOCAB_SIZE, embedding_dim, input_length=MAX_LEN),
        Dropout(dropout_rate),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\n--- 1D CNN Model Summary ---")
    model.summary()
    return model

def build_bilstm_model(embedding_dim=128, dropout_rate=0.3, lstm_units=64):
    K.clear_session()
    model = Sequential([
        Embedding(VOCAB_SIZE, embedding_dim, input_length=MAX_LEN),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units)),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\n--- Bi-LSTM Model Summary ---")
    model.summary()
    return model

# --- Main Training and Evaluation Pipeline ---
def run_pipeline():
    global MAX_LEN

    # 1. PREPARE DATA
    print("--- Loading and Preparing Data ---")
    file_path = 'datasets.csv'
    try:
        df = pd.read_csv(file_path) # read the dataset first
    except FileNotFoundError:
        print(f"\nError: '{file_path}' not found. Please place it in the same folder.")
        return

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df.dropna(subset=['text', 'class'], inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text) # to standarize it
    df['label'] = df['class'].map({'depression': 1, 'non-depression': 0}) # encode label to number values 

    texts = df['cleaned_text'].values
    labels = df['label'].values

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>") # to convert words to numbers
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    MAX_LEN = max(len(x) for x in sequences) if texts.size > 0 else 50

    X = pad_sequences(sequences, padding='post', maxlen=MAX_LEN)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data prepared. Max sequence length: {MAX_LEN}")

    '''
    neural network only understands numbers, so we gotta convert the text into a standardized numerical format.
    We use tokenize to convert words to numbers, pad for the fixed inputs required by the model, and split the data between
    test and train to get an unbiased evaluation on the true performance
    '''

    # 2. TRAIN AND EVALUATE CNN
    cnn_model = build_cnn_model()
    print("\n--- Training 1D CNN Model ---")
    cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    print("\n--- Evaluating 1D CNN Model ---")
    y_pred_proba_cnn = cnn_model.predict(X_test)
    y_pred_cnn = (y_pred_proba_cnn > 0.5).astype("int32")

    accuracy_cnn = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
    precision_cnn = precision_score(y_test, y_pred_cnn)
    recall_cnn = recall_score(y_test, y_pred_cnn)
    f1_cnn = f1_score(y_test, y_pred_cnn)

    print("+-----------------+----------------+")
    print("| CNN Metric      | Value          |")
    print("+-----------------+----------------+")
    print(f"| Accuracy        | {accuracy_cnn*100:<15.2f}% |")
    print(f"| Precision       | {precision_cnn:<15.4f} |")
    print(f"| Recall          | {recall_cnn:<15.4f} |")
    print(f"| F1-Score        | {f1_cnn:<15.4f} |")
    print("+-----------------+----------------+")

    # 3. TRAIN AND EVALUATE BI-LSTM
    bilstm_model = build_bilstm_model()
    print("\n--- Training Bi-LSTM Model ---")
    bilstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    print("\n--- Evaluating Bi-LSTM Model ---")
    y_pred_proba_bilstm = bilstm_model.predict(X_test)
    y_pred_bilstm = (y_pred_proba_bilstm > 0.5).astype("int32")

    accuracy_bilstm = bilstm_model.evaluate(X_test, y_test, verbose=0)[1]
    precision_bilstm = precision_score(y_test, y_pred_bilstm)
    recall_bilstm = recall_score(y_test, y_pred_bilstm)
    f1_bilstm = f1_score(y_test, y_pred_bilstm)

    print("+-----------------+----------------+")
    print("| Bi-LSTM Metric  | Value          |")
    print("+-----------------+----------------+")
    print(f"| Accuracy        | {accuracy_bilstm*100:<15.2f}% |")
    print(f"| Precision       | {precision_bilstm:<15.4f} |")
    print(f"| Recall          | {recall_bilstm:<15.4f} |")
    print(f"| F1-Score        | {f1_bilstm:<15.4f} |")
    print("+-----------------+----------------+")

    '''
    model fit is to start training process on data
    model predict is to get unseen test data's prediction
    precision, recall, f1 is for metrics
    model save to save the keras model 

    we fit the model to teach it patterns in the data. we then predict on the test set to get a true evaluation of the performance
    
    '''

    # 4. SAVE FINAL ASSETS (Optional, but good practice)
    print("\n--- Saving final models and preprocessing info ---")
    cnn_model.save('cnn_model.keras')
    bilstm_model.save('bilstm_model.keras')

    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    config = {'max_len': MAX_LEN}
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f)

    print("--- All assets saved successfully. ---")


if __name__ == '__main__':
    run_pipeline()