# главная страница веб приложения
import streamlit as st
from pathlib import Path

pages_dir = Path('pages')

st.title("Computer vision project • Faster-RCNN team")

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к странице:", ["Лица / Yolov5", "Ветряки / Yolov5", "Лес / Unet"])


if selection == "Лица / Yolov5":
    marina_file = pages_dir / 'marina' / 'main.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location("marina", str(marina_file))
    marina_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(marina_module)

elif selection == "Ветряки / Yolov5":
    olga_file = pages_dir / 'olga' / 'main.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location("olga", str(olga_file))
    olga_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(olga_module)

elif selection == "Лес / Unet":
    danila_file = pages_dir / 'danila' / 'danila.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location("danila", str(danila_file))
    danila_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(danila_module)

