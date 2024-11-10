# pdf_processor.py
import pytesseract
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from pdf2image import convert_from_path
import PyPDF2
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class PDFProcessor2:
    def __init__(self, file):
        self.file = file
        self.pages_text = []
        self.reader = None

        try:
            pdf_bytes = self.file.read()
            self.reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            self.file.seek(0)  # Reset file pointer after reading
        except Exception as e:
            logger.error(f"Error opening with PyPDF2: {e}")
            self.reader = None

    def extract_text_by_page(self):
        if not self.reader:
            raise ValueError("PDF document is not initialized.")

        self.pages_text = []
        for i, page in enumerate(self.reader.pages):
            logger.info(f"Processing page {i + 1}/{len(self.reader.pages)}...")
            extracted_text = page.extract_text()
            if extracted_text:
                logger.debug(f"Text extracted from page {i + 1}: {extracted_text[:500]}...\n")
                self.pages_text.append(extracted_text)
            else:
                logger.info(f"No text found on page {i + 1}, switching to OCR...")
                ocr_text = self.extract_text_with_ocr_per_page(i)
                self.pages_text.append(ocr_text)

        return self.pages_text

    def extract_text_with_ocr_per_page(self, page_number):
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(self.file.read())
            temp_pdf_path = temp_pdf.name

        try:
            images = convert_from_path(temp_pdf_path, first_page=page_number + 1, last_page=page_number + 1)
            ocr_text = ""

            for image in images:
                logger.info(f"Processing OCR for page {page_number + 1}...")
                open_cv_image = np.array(image)
                processed_image = self.preprocess_image_for_ocr(open_cv_image)
                ocr_text += pytesseract.image_to_string(processed_image)

            return ocr_text
        finally:
            # Eliminar el archivo temporal
            os.remove(temp_pdf_path)

    def extract_text_with_ocr(self):
        # Similar a extract_text_with_ocr_per_page, pero para todas las páginas
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(self.file.read())
            temp_pdf_path = temp_pdf.name

        ocr_text = []
        try:
            images = convert_from_path(temp_pdf_path)
            for i, image in enumerate(images):
                logger.info(f"Processing OCR for page {i + 1}/{len(images)}...")
                open_cv_image = np.array(image)
                processed_image = self.preprocess_image_for_ocr(open_cv_image)
                ocr_text.append(pytesseract.image_to_string(processed_image))

            return ocr_text
        finally:
            os.remove(temp_pdf_path)

    def preprocess_image_for_ocr(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
        kernel = np.ones((1, 1), np.uint8)
        dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)
        return dilated_image
