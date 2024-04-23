import requests
import os
import zipfile

def download_dataset(url, save_path, extract_path):
    if not os.path.exists(save_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)
        print("Download complete.")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")


url = "vqa_dataset.zip"  # This is a placeholder URL
save_path = "vqa_dataset.zip"
extract_path = "./data"

download_dataset(url, save_path, extract_path)



import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Tokenize and pad text
def tokenize_and_pad_texts(texts, num_words=10000, max_len=20):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, tokenizer

# Example usage
image_path = "./data/sample_image.jpg"
processed_image = load_and_preprocess_image(image_path)

questions = ["What is in the image?", "How many objects are there?"]
processed_questions, tokenizer = tokenize_and_pad_texts(questions)

print("Processed Image Shape:", processed_image.shape)
print("Processed Questions Shape:", processed_questions.shape)



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50

# Parameters
vocab_size = 10000  # Adjust as per your tokenizer's vocabulary size
embedding_dim = 256
max_question_length = 20  # Adjust based on your dataset preprocessing
num_classes = 1000  # Adjust based on the number of answer classes in your dataset

# Image feature extractor model
def build_image_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

# Question feature extractor model
def build_text_model():
    input_seq = Input(shape=(max_question_length,))
    embedded_seq = Embedding(vocab_size, embedding_dim, input_length=max_question_length)(input_seq)
    lstm_out = LSTM(128)(embedded_seq)
    return Model(inputs=input_seq, outputs=lstm_out)

# Combined VQA Model
def build_vqa_model(image_model, text_model):
    combined_input = Concatenate()([image_model.output, text_model.output])
    x = Dense(512, activation='relu')(combined_input)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[image_model.input, text_model.input], outputs=x)

# Build and compile the model
image_model = build_image_model()
text_model = build_text_model()
vqa_model = build_vqa_model(image_model, text_model)

vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vqa_model.summary()



# Dummy variables for illustration
train_images = []  # Should be numpy arrays from your preprocessing step
train_questions = []  # Should be tokenized and padded sequences
train_answers = []  # Should be one-hot encoded answers

vqa_model.fit([train_images, train_questions], train_answers, epochs=10, batch_size=64, validation_split=0.1)

test_images = []  # Test dataset images
test_questions = []  # Test dataset questions
test_answers = []  # Test dataset answers

loss, accuracy = vqa_model.evaluate([test_images, test_questions], test_answers)
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

