import cv2
import torch

# 加载图像
image_path = 'own_datas/test/water.jpg'
image = cv2.imread(image_path)

# 打印原始图像的形状
print("Original image shape:", image.shape[2]) # 管道数输出3 表示是彩色图象 输出1表示是灰色图像
# 如果 image.shape[2] 的值为 3，表示图像是一个彩色图像（RGB 或 BGR 格式）。这意味着每个像素由三个颜色通道组成，分别是红色、绿色和蓝色（RGB）或蓝色、绿色和红色（BGR）。
# 如果 image.shape[2] 的值为 1，表示图像是灰度图像。这意味着每个像素只有一个灰度值，表示图像的亮度。
