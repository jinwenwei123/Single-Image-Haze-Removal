import cv2
import numpy as np

# 导入原始图像
image = cv2.imread("./R.jpg")


# 计算暗通道图像
def dark_channal(image):
    blue, green, red = cv2.split(image)  # 分离原始图像红、绿、蓝三通道
    dark_image = cv2.min(cv2.min(blue, green), red)  # 计算以像素为单位的暗通道图像
    erode_kernel = np.ones((15, 15), dtype=np.uint8)  # 定义15*15的腐蚀操作的卷积核
    dark_image = cv2.erode(dark_image, erode_kernel)  # 对暗通道图像进行腐蚀，放大暗通道的特征
    return dark_image


# 估计大气光数值 A
dark_image = dark_channal(image)
dark_px = dark_image.shape[0] * dark_image.shape[1]  # 计算暗通道图像像素点总和
num_px = int(max(dark_px // 1000, 1))  # 计算dark_px里0.1%像素的值，和 1 相比取最大值，保证至少取得一个像素
dark_arr = dark_image.reshape(dark_px)  # 将暗通道图像拉直
image_arr = image.reshape(dark_px, 3)  # 将原始图像每个通道拉直
index = dark_arr.argsort()  # 暗通道数组像素值从小到大的索引
index = index[dark_px - num_px:]  # 将索引切片，选取最后0.1%像素的索引

a_sum = np.zeros((1, 3))  # 储存累加的大气光值
for i in index[:-1]:  # 去除最亮的像素点，提高精准度
    a_sum += image_arr[i]

A = a_sum / num_px  # 大气光估值

# 估计传输率 t
nor = np.empty(image.shape, dtype=np.float64)  # 定义归一化图像
for i in range(3):  # 将原始图像每一个通道分别归一化
    nor[:, :, i] = image[:, :, i] / (A[0, i] + 1e-6)

te = 1 - 0.95 * dark_channal(nor)  # 根据公式计算出传输率估值te，参数设置为0.95

# 使用引导滤波器对传输率t进行细化
ks=50
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将原始图片转换为灰度图
gray_image = np.float64(gray_image) / 255  # 进行归一化

mean_I = cv2.boxFilter(gray_image, cv2.CV_64F, (ks, ks))  # 引导图像均值滤波
mean_p = cv2.boxFilter(te, cv2.CV_64F, (ks, ks))  # 待处理图像均值滤波

mean_Ip = cv2.boxFilter(gray_image * te, cv2.CV_64F, (ks, ks))  # 计算协方差
cov_Ip = mean_Ip - mean_I * mean_p

var_I = cv2.boxFilter(gray_image * gray_image, cv2.CV_64F, (ks, ks)) - mean_I * mean_I  # 计算引导图像的方差

a = cov_Ip / (var_I + 0.0001)  # 线性系数 a 和 b
b = mean_p - a * mean_I

mean_a = cv2.boxFilter(a, cv2.CV_64F, (ks, ks))  # 对系数 a 和 b 进行均值滤波
mean_b = cv2.boxFilter(b, cv2.CV_64F, (ks, ks))

t = mean_a * gray_image + mean_b  # 输出细化后的传输率 t

# 输出无雾霾的图像
processed_image = np.empty(image.shape, image.dtype)  # 创建去雾霾处理后的图像
t = cv2.max(t, 0.1)  # 为 t 设定最小值，防止不稳定

for i in range(0, 3):  # 根据论文推导出的恢复公式，计算出去雾霾处理后的图片
    processed_image[:, :, i] = ((image[:, :, i] - A[0, i]) / t + A[0, i]).clip(0, 255)  # 将像素值限制在(0,255)，防止出现颜色失真

# 打印原始图像"raw_iamge"，暗通道图像"dark_iamge"，传输率图像"t"和去雾霾处理后的图像"processed_image"
cv2.imshow("raw_iamge", image)
cv2.imshow("dark_iamge", dark_image)
cv2.imshow("t", t)
cv2.imshow("processed_image", processed_image)
cv2.waitKey()
cv2.destroyAllWindows()
