import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
image_path = 'daun.jpeg'  
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Deteksi tepi menggunakan operator Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)

# Deteksi tepi menggunakan operator Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.abs(laplacian) / np.abs(laplacian).max() * 255)

# Deteksi tepi menggunakan deteksi Canny
canny_edges = cv2.Canny(gray, 100, 200)

# Segmentasi gambar
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
segmented = cv2.bitwise_and(gray, gray, mask=binary)

# Menampilkan hasil
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel - Horizontal')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel - Vertical')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel - Magnitude')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.figure(figsize=(6, 6))
plt.imshow(segmented, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
