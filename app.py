import streamlit as st
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

st.title('MNIST Image Viewer')

number = st.number_input("Enter a number (0-9)", min_value=0, max_value=9, value=0, step=1)

if st.button('Show Image'):
    # Filter for the selected number
    filtered_indices = [i for i, label in enumerate(train_labels) if label == number]

    if filtered_indices:
        # Display the first image found with the selected number
        selected_index = filtered_indices[0]
        st.image(train_images[selected_index], width=150, caption=f"Image of {number}")
    else:
        st.write(f"No images found for the number {number} in the dataset.")
********************************************************************************************************************************


import streamlit as st
import tensorflow as tf

# Function to create a simple model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

st.title('Train a Simple Model')

# Input for the number of epochs
num_epochs = st.number_input("Enter the number of epochs", min_value=1, value=5, step=1)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Train the model based on user input
if st.button('Train Model'):
    model = create_model()
    model.fit(train_images, train_labels, epochs=num_epochs)
    st.write(f"Model trained for {num_epochs} epochs.")
