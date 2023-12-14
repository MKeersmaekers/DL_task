import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to download images
def download_images(categories):
    for category in categories:
        os.makedirs(category, exist_ok=True)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        for page in range(1, 3):
            base_url = f'https://www.gettyimages.be/search/2/image?phrase={category}&page={page}&numberofpeople=none'
            response = requests.get(base_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img', {'class': 'BLA_wBUJrga_SkfJ8won'})
            urls = [img['src'] for img in img_tags if 'src' in img.attrs]

            for i, url in enumerate(urls[:100]):
                response = requests.get(url, headers=headers)
                with open(f'{category}/{category}_{page*100+i}.jpg', 'wb') as f:
                    f.write(response.content)

# Function to display images
def show_images(categories):
    for category in categories:
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        fig.suptitle(category)
        for i, img in enumerate(os.listdir(category)[:5]):
            img_path = os.path.join(category, img)
            axs[i].imshow(mpimg.imread(img_path))
            axs[i].axis('off')
        st.pyplot(fig)

# Function to create and compile the model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_generator, validation_generator, epochs):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return history

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    st.pyplot()

# Function to evaluate the model
def evaluate_model(model, test_generator):
    test_labels = test_generator.classes
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, predicted_labels)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot()

def main():
    st.title('Image Classification Streamlit App')

    # Download images
    if st.button('Download Images'):
        download_images(categories)
        st.success('Images downloaded successfully!')

    # Visualize images
    if st.button('Visualize Images'):
        show_images(categories)

    # Train and evaluate the model
    if st.button('Train and Evaluate Model'):
        model = create_model()

        # Create data generators
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = datagen.flow_from_directory(directory='.', target_size=(150, 150), batch_size=32,
                                                     class_mode='categorical', subset='training')
        validation_generator = datagen.flow_from_directory(directory='.', target_size=(150, 150), batch_size=32,
                                                          class_mode='categorical', subset='validation')

        # Train the model
        epochs = st.slider('Select the number of epochs:', min_value=1, max_value=20, value=10)
        history = train_model(model, train_generator, validation_generator, epochs)

        # Plot training history
        plot_training_history(history)

        # Evaluate the model
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(directory='.', target_size=(150, 150), batch_size=32,
                                                          class_mode='categorical')

        evaluate_model(model, test_generator)

if __name__ == '__main__':
    main()
