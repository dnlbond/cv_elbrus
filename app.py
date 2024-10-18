# главная страница веб приложения
import streamlit as st
from pathlib import Path

pages_dir = Path('pages')

st.title("Computer vision project • Faster-RCNN team")

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к странице:", ["Лица / Yolov5", "Ветряки / Yolov5", "Лес / Unet"])


if selection == "Лица / Yolov5":
    import pages.marina
    pages.marina.main()
elif selection == "Ветряки / Yolov5":
    import pages.olga
    pages.olga.main()
elif selection == "Лес / Unet":
    danila_file = pages_dir / 'danila' / 'danila.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location("danila", str(danila_file))
    danila_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(danila_module)

