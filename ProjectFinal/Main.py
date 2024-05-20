import sys
import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import streamlit as st
sys.path.insert(0, ".")
from EditImage import enhancement_range, filter_range
import Predict
import yolov8
import Filter


# Function to handle image upload
def handle_image_upload():
    uploaded_file = st.file_uploader("Upload Art", key="file_uploader")
    if uploaded_file is not None:
        try:
            return Image.open(uploaded_file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")

# Function to handle image URL input
def handle_image_url_input():
    image_url = st.text_input("Image URL", key="image_url")
    if st.button("Submit"):
        if image_url:
            try:
                response = requests.get(image_url)
                return Image.open(BytesIO(response.content))
            except:
                st.error("The URL does not seem to be valid.")

# Convert RGBA to RGB if necessary
def convert_rgba_to_rgb(img):
    n_dims = np.array(img).shape[-1]
    if n_dims == 4:
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        return background
    return img

# Apply image enhancements
def apply_image_enhancements(img, enhancement_factor_dict):
    for cat in enhancement_range.keys():
        img = getattr(ImageEnhance, cat)(img)
        img = img.enhance(enhancement_factor_dict[cat])
    return img

def apply_image_filter(img, filter_dict):
    # Chuyển hình ảnh PIL sang mảng numpy
    img_array = np.array(img)

    # Xử lý hình ảnh dựa trên các bộ lọc trong filter_dict
    for filter_type, value in filter_dict.items():
        if filter_type == "Median":
            # Áp dụng bộ lọc trung vị
            img_array = Filter.median_filter(img_array, value)
        elif filter_type == "Average":
            # Áp dụng bộ lọc trung bình
            img_array = Filter.average_filter(img_array, value)
        elif filter_type == "Gauss":
            # Áp dụng bộ lọc trung bình
            img_array = Filter.gaussian_filter(img_array, value)

    # Chuyển mảng numpy đã lọc trở lại thành đối tượng hình ảnh PIL và trả về
    return Image.fromarray(img_array)
# Định nghĩa một lớp Transformer để xử lý dữ liệu video

# Display artwork
def display_artwork(image):
    with st.expander("Picture", expanded=True):
        st.image(image, use_column_width=True)

# Main function
def main():

    # Set up Streamlit app
    st.set_page_config(
        page_title="Vegetables Predict",
        page_icon="🥦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Thiết lập CSS cho trang web
    custom_css = """
        <style>
            body {
                font-family: Arial, sans-serif; /* Chọn font chữ mặc định */
            }
            .stTab {
                font-family: 'Monsteratt', sans-serif; /* Font cho các tab */
                font-weight: extra bold; /* Độ đậm cho các tab */
                color: #333; /* Màu chữ cho các tab */
                background-color: #4CAF50; /* Màu nền cho các tab */
                padding: 8px 16px;
                border-radius: 5px 5px 0 0;
                cursor: pointer;
            }
            .stTab:hover {
                background-color: #45a049; /* Màu nền khi di chuột qua các tab */
            }
            .stTab.stTabActive {
                color: white; /* Màu chữ khi tab đang active */
            }
        </style>
    """

    # Áp dụng CSS cho trang web
    st.markdown(custom_css, unsafe_allow_html=True)

    #Ảnh header
    st.image('media/home.png')

    # Áp dụng CSS cho trang web
    st.markdown(custom_css, unsafe_allow_html=True)

    st.sidebar.image('media/TitleEdit.png')
    # Sử dụng Markdown và CSS inline để thiết lập màu nền cho tiêu đề trong sidebar
    st.sidebar.markdown("---")
    # Display other Streamlit
    with st.sidebar.expander("Members in team"):
        st.caption("Trần Văn Bảo Duy")
        st.caption("Bùi Đặng Thùy Thương")
        st.caption("Lê Xuân Bách")
        st.caption("Đinh Thị Thúy Quỳnh")

    st.sidebar.markdown("---")

    st.sidebar.image('media/TitleAdjustment.png')


    # Image enhancement
    enhancement_factor_dict = {}
    with st.sidebar.expander("Image Enhancements", expanded=False):
        for cat in enhancement_range.keys():
            enhancement_factor_dict[cat] = st.slider(f"{cat} Enhancement",
                                                      value=1.,
                                                      min_value=enhancement_range[cat][0],
                                                      max_value=enhancement_range[cat][1],
                                                      step=enhancement_range[cat][2],
                                                      key=f"{cat}_enhancement")

    st.sidebar.image('media/Filter.png')
    filter_dict = {}
    with st.sidebar.expander("Image Filter", expanded=False):
        for cat in filter_range.keys():
            filter_dict[cat] = st.slider(f"{cat} Filter",
                                         value=1,
                                         min_value=filter_range[cat][0],
                                         max_value=filter_range[cat][1],
                                         step=filter_range[cat][2],
                                         key=f"{cat}_filter")
    # Thiết lập các tab
    camera_tab, upload_tab, url_tab = st.tabs(["**Camera**", "**Upload**", "**Image URL**"])

    # Thiết lập màu nền và màu chữ cho các tab
    tab_styles = """
    <style>
        .green-tab {
            background-color: #0E810A;
            color: white;
            padding: 8px 16px;
            border-radius: 5px 5px 0 0;
            cursor: pointer;
        }
        .green-tab:hover {
            background-color: #45a049;
        }
    </style>
    """

    # Áp dụng CSS cho các tab
    st.markdown(tab_styles, unsafe_allow_html=True)

    # Hiển thị các tab
    with camera_tab:
        st.markdown("<div class='green-tab'>Gallery</div>", unsafe_allow_html=True)
        # Thêm nội dung vào tab Gallery nếu cần
        st.markdown("---")
        if st.button('Use camera'):
            yolov8.count_fruits_on_camera()

    with upload_tab:
        st.markdown("<div class='green-tab'>Upload</div>", unsafe_allow_html=True)
        # Thêm nội dung vào tab Upload nếu cần
        file = st.file_uploader("", key="file_uploader")
        if file is not None:
            try:
                img = Image.open(file)
            except:
                st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
        if st.session_state.get("image_url") not in ["", None]:
            st.warning("To use the file uploader, remove the image URL first.")

    with url_tab:
        st.markdown("<div class='green-tab'>Image URL</div>", unsafe_allow_html=True)
        # Thêm nội dung vào tab Image URL nếu cần
        url_text = st.empty()
        url = url_text.text_input("", key="image_url")

        if url != "":
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            except:
                st.error("The URL does not seem to be valid.")

    if "img" in locals():
        img = convert_rgba_to_rgb(img)
        img = apply_image_enhancements(img, enhancement_factor_dict)
        img = apply_image_filter(img, filter_dict)
        display_artwork(img)

    if st.button("Predict"):
        # Convert image to JPEG format
        img_jpg = img.convert('RGB')
        img_jpg.save('image.jpg', format='JPEG')
        Predict.predict_image(img)

# Call the main function
if __name__ == "__main__":
    main()
