import cv2
import numpy as np

def median_filter(img,size):
        h, w, channels = img.shape
        B = np.zeros_like(img, dtype=np.uint8)
        for i in range(0,h-size+1):
                for j in range(0,w-size+1):
                    sImage = img[i:i+size,j:j+size]
                    for c in range(channels):
                        B[i,j,c]=np.median(sImage[:,:,c])
        imgB = B[0:h-size+1,0:w-size+1]
        return imgB
def average_filter(img, size):
    h, w, channels = img.shape
    B = np.zeros_like(img, dtype=np.uint8)

    # Lặp qua mỗi pixel trong ảnh mà không tính các pixel ở biên nếu cần
    padding = size // 2
    for i in range(padding, h-padding):
        for j in range(padding, w-padding):
            # Tính giá trị trung bình cho từng kênh màu trong cửa sổ kernel
            for c in range(channels):
                # Trích xuất cửa sổ kernel chứa các pixel
                window = img[i-padding:i+padding+1, j-padding:j+padding+1, c]
                # Tính giá trị trung bình của cửa sổ kernel
                average_value = np.mean(window)
                # Gán giá trị trung bình này vào ảnh kết quả
                B[i, j, c] = int(average_value)

    # Trả về phần của ảnh đã lọc, loại bỏ padding nếu cần
    img_filtered = B[padding:h-padding, padding:w-padding]
    return img_filtered

def gaussian_filter(img, kernel_size, sigma=0):
    # Áp dụng bộ lọc Gaussian
    # Nếu sigma = 0, OpenCV sẽ tính giá trị sigma dựa trên kernel_size
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

