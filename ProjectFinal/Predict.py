import streamlit as st
import tensorflow as tf
import numpy as np
import yolov8


def predict_image(image):
    if image is not None:
        st.image(image, caption="Uploaded Image", width=200)
        result = yolov8.detect_fruit_in_image(image)
        # Thêm CSS để tùy chỉnh kiểu cho phần kết quả
        st.markdown("""
        <style>
        .result {
            color: green; /* Đổi màu chữ thành màu xanh */
            font-size: 30px; /* Đổi kích thước chữ thành 20px */
            font-weight: bold; /* In đậm */
        }
        </style>
        """, unsafe_allow_html=True)

        # Hiển thị kết quả với kiểu được tùy chỉnh
        st.markdown(f"<p class='result'>{yolov8.format_result(result)}</p>", unsafe_allow_html=True)
