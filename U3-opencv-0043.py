import cv2
# Sebastian Farfan NC = 0043
# Nombre del archivo (cambia a .png si aplica)
ruta_imagen = 'Capricorn.jpg'
image = cv2.imread(ruta_imagen)

if image is None:
    raise FileNotFoundError(f"No se encontró '{ruta_imagen}'. Verifica el nombre y la extensión.")

# 1. Mostrar imagen original
cv2.imshow('Capricorn', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Capricorn_gray.jpg', gray_image)

# 3. Redimensionar (300x200)
resized_image = cv2.resize(image, (300, 200))
cv2.imwrite('Capricorn_resized.jpg', resized_image)

# 4. Rotar 90 grados
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(image, matrix, (w, h))
cv2.imwrite('Capricorn_rotated.jpg', rotated_image)

# 5. Desenfoque Gaussiano
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite('Capricorn_blurred.jpg', blurred_image)

# 6. Detección de bordes Canny
edges = cv2.Canny(gray_image, 100, 200)
cv2.imwrite('Capricorn_edges.jpg', edges)

# 7. Dibujar rectángulo y texto
annotated_image = image.copy()
cv2.rectangle(annotated_image, (50, 50), (200, 200), (255, 0, 0), 3)
cv2.putText(annotated_image, 'Capricorn', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('Capricorn_annotated.jpg', annotated_image)

# 8. Umbralización binaria
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('Capricorn_threshold.jpg', thresh_image)

# 9. Detección de contornos
contours_image = image.copy()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite('Capricorn_contours.jpg', contours_image)

print("Listo, todas las imágenes fueron generadas por Sebastian Farfan NC = 0043.")