import cv2
import pytesseract
import tempfile
from io import BytesIO

class OCRImage:
    def process_image(self, image: BytesIO):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(image.getbuffer())
            file_path = tmp.name

        # Load the image
        image = cv2.imread(file_path)

        # Zoom the image
        scale_percent = 200  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Preprocessing the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Using Gaussian blurring

        # Applying adaptive threshold
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Applying dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.dilate(gray, kernel, iterations=1)

        # Applying OCR using Tesseract
        text = pytesseract.image_to_string(gray, lang='eng')

        # Return the extracted text without spaces
        return text.strip()


