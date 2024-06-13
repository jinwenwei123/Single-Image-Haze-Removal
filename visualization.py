"""
可视化处理
"""

import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 读取图像
f = "01"
image = cv2.imread("images/" + f + "/image.jpeg")
dark_image = cv2.imread("images/" + f + "/dark_image.jpeg")
t = cv2.imread("images/" + f + "/t.jpeg")
processed_image = cv2.imread("images/" + f + "/processed_image.jpeg")

# 1.对照图组输出
fig, axs = plt.subplots(1, 4, figsize=(20, 10))

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(dark_image, cmap='gray')
axs[1].set_title('Dark Channel')
axs[1].axis('off')

axs[2].imshow(t, cmap='gray')
axs[2].set_title('Transmission Map')
axs[2].axis('off')

axs[3].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
axs[3].set_title('Dehazed Image')
axs[3].axis('off')

plt.show()

# 2.峰值信噪比（PSNR）和结构相似性指数（SSIM）
original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dehazed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

psnr_value = psnr(original_gray, dehazed_gray)
ssim_value = ssim(original_gray, dehazed_gray)

print(f'PSNR: {psnr_value}')
print(f'SSIM: {ssim_value}')
