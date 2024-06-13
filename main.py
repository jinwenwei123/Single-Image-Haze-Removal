"""
图像去雾霾算法
"""

import cv2
import numpy as np


# 计算暗通道图像
def dark_channal(image):
    blue, green, red = cv2.split(image)  # 分离原始图像红、绿、蓝三通道
    dark_image = cv2.min(cv2.min(blue, green), red)  # 计算以像素为单位的暗通道图像
    erode_kernel = np.ones((15, 15), dtype=np.uint8)  # 定义15*15的腐蚀操作的卷积核
    dark_image = cv2.erode(dark_image, erode_kernel)  # 对暗通道图像进行腐蚀，放大暗通道的特征
    return dark_image


# 图像去雾霾处理
def image_process(image):
    # 估计大气光数值 A
    dark_image = dark_channal(image)
    dark_px = dark_image.shape[0] * dark_image.shape[1]  # 计算暗通道图像像素点总和
    num_px = int(max(dark_px // 1000, 10))  # 计算dark_px里0.1%像素的值，和 1 相比取最大值，保证至少取得一个像素
    dark_arr = dark_image.reshape(dark_px)  # 将暗通道图像拉直
    image_arr = image.reshape(dark_px, 3)  # 将原始图像每个通道拉直
    index = dark_arr.argsort()  # 暗通道数组像素值从小到大的索引
    index = index[dark_px - num_px:]  # 将索引切片，选取最后0.1%像素的索引

    a_sum = np.zeros((1, 3))  # 储存累加的大气光值
    for i in index[:]:  # 去除最亮的像素点，提高精准度
        a_sum += image_arr[i]

    A = a_sum / num_px  # 大气光估值

    # 估计传输率 t
    nor = np.empty(image.shape, dtype=np.float64)  # 定义归一化图像
    for i in range(3):  # 将原始图像每一个通道分别归一化
        nor[:, :, i] = image[:, :, i] / (A[0, i] + 1e-6)

    te = 1 - 0.95 * dark_channal(nor)  # 根据公式计算出传输率估值te，参数设置为0.95

    # 使用双边滤波器对传输率t进行细化
    d = 15  # 双边滤波的直径
    sigmaColor = 75  # 双边滤波的颜色空间标准差
    sigmaSpace = 75  # 双边滤波的坐标空间标准差
    t = cv2.bilateralFilter(te.astype(np.float32), d, sigmaColor, sigmaSpace)

    # 输出无雾霾的图像
    processed_image = np.empty(image.shape, image.dtype)  # 创建去雾霾处理后的图像
    t = cv2.max(t, 0.3)  # 为 t 设定最小值，防止不稳定

    for i in range(0, 3):  # 根据论文推导出的恢复公式，计算出去雾霾处理后的图片
        processed_image[:, :, i] = ((image[:, :, i] - A[0, i]) / t + A[0, i]).clip(0, 255)  # 将像素值限制在(0,255)，防止出现颜色失真
    return dark_image, t, processed_image


if __name__ == '__main__':
    f = "04"
    # 读取图像
    image = cv2.imread("images/" + f + "/image.jpeg")

    # 打印原始图像"raw_iamge"，暗通道图像"dark_iamge"，传输率图像"t"和去雾霾处理后的图像"processed_image"
    cv2.imshow("raw_iamge", image)
    dark_image, t, processed_image = image_process(image)
    cv2.imshow("dark_image", dark_image)
    cv2.imshow("t", t)
    cv2.imshow("processed_image", processed_image)
    cv2.imwrite("images/" + f + "/dark_image.jpeg", dark_image)
    cv2.imwrite("images/" + f + "/t.jpeg", (t * 255).astype(np.uint8))
    cv2.imwrite("images/" + f + "/processed_image.jpeg", processed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
