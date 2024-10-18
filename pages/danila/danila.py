import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2 # тест

# Загрузка модели Keras
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/danila/new_model.keras')
    return model

model = load_model()

# Главная страница
st.title("Семантическая сегментация снмиков леса с помощью модели Unet")

# Раздел для загрузки файлов
st.header("Загрузка изображений для распознавания")
uploaded_files = st.file_uploader("Выберите изображения", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

def predict_and_display(image):
    # Предобработка изображения
    image = image.resize((128, 128))  # Измените размер на тот, который использовался при обучении модели
    image = image.convert('RGB')  # Конвертация в RGB, если изображение не имеет 3 каналов
    image_array = np.array(image) / 255.0  # Нормализация

    # Проверка количества каналов
    if image_array.shape[-1] != 3:
        st.error("Input image must have 3 channels (RGB).")
        return

    image_array = np.expand_dims(image_array, axis=0)

    # Предсказание
    prediction = model.predict(image_array)

    # Преобразование предсказания в бинарную маску
    predicted_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
    
    # Масштабирование маски до 0-255 для корректного отображения
    predicted_mask = predicted_mask * 255

    # Отображение исходного изображения и маски
    st.image(image, caption='Оригинал')
    st.image(predicted_mask, caption='Маска предсказания', clamp=True)
    # Если маска отображается некорректно, можно использовать:
    # st.image(predicted_mask, caption='Predicted Mask', clamp=True, channels="L")

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        predict_and_display(image)

# Раздел для загрузки по ссылке
st.header("Загрузка изображений по URL")
url = st.text_input("Вставьте ссылку на изображение")

if url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        predict_and_display(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Раздел с информацией о модели
st.header("Информация о модели Unet")
   
# Информация о процессе обучения
st.text("Количество эпох обучения: 10")

summary_path = "/home/dnl/ds_bootcamp/cv_elbrus/Images/danila/model_summary.jpg"
summary_image = Image.open(summary_path)
st.image(summary_image, use_column_width=True)


# Информация о процессе обучения
st.text("Ход обучения:")
learning_path = "/home/dnl/ds_bootcamp/cv_elbrus/Images/danila/learning.jpg"
learning_image = Image.open(learning_path)
st.image(learning_image, use_column_width=True)

# Информация о процессе обучения
pr_path = "/home/dnl/ds_bootcamp/cv_elbrus/Images/danila/PR-curve.jpg"
pr_image = Image.open(pr_path)
st.image(pr_image, use_column_width=True)


# Информация о процессе обучения
st.text("Примеры обучения:")
example_path = "/home/dnl/ds_bootcamp/cv_elbrus/Images/danila/learning-examples.jpg"
example_image = Image.open(example_path)
st.image(example_image, use_column_width=True)



