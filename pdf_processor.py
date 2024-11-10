import pytesseract
from PIL import Image
from io import BytesIO
import cv2
import easyocr
import numpy as np
from pdf2image import convert_from_path
import PyPDF2


class PDFProcessor:
    def __init__(self, file):
        self.file = file
        self.pages_text = []
        self.reader = None

        try:
            pdf_bytes = self.file.read()
            self.reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        except Exception as e:
            print(f"Error opening with PyPDF2: {e}")
            self.reader = None

    def extract_text_by_page(self):
        if not self.reader:
            raise ValueError("PDF document is not initialized.")

        self.pages_text = []
        try:
            for i, page in enumerate(self.reader.pages):
                print(f"Processing page {i + 1}/{len(self.reader.pages)}...")
                extracted_text = page.extract_text()
                if extracted_text:
                    print(f"Text extracted from page {i + 1}: {extracted_text[:5000]}...\n")
                    self.pages_text.append(extracted_text)
                else:
                    print(f"No text found on page {i + 1}, switching to OCR...")
                    ocr_text = self.extract_text_with_ocr_per_page(i)
                    self.pages_text.append(ocr_text)
        except Exception as e:
            print(f"Error encountered: {e}, switching to OCR for entire document...")
            self.pages_text = self.extract_text_with_ocr()

        return self.pages_text

    def extract_text_with_ocr_per_page(self, page_number):
        pdf_bytes = self.file.read()
        images = convert_from_path(BytesIO(pdf_bytes), first_page=page_number + 1, last_page=page_number + 1)
        ocr_text = ""

        for image in images:
            print(f"Processing OCR for page {page_number + 1}...")
            open_cv_image = np.array(image)
            processed_image = self.preprocess_image_for_ocr(open_cv_image)
            ocr_text += pytesseract.image_to_string(processed_image)

        return ocr_text

    def extract_text_with_ocr(self):
        pdf_bytes = self.file.read()
        images = convert_from_path(BytesIO(pdf_bytes))
        ocr_text = []

        for i, image in enumerate(images):
            print(f"Processing OCR for page {i + 1}/{len(images)}...")
            open_cv_image = np.array(image)
            processed_image = self.preprocess_image_for_ocr(open_cv_image)
            ocr_text.append(pytesseract.image_to_string(processed_image))

        return ocr_text

    def preprocess_image_for_ocr(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
        kernel = np.ones((1, 1), np.uint8)
        dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)
        return dilated_image

    def extract_text_with_easyocr(self):
        reader = easyocr.Reader(['en'])
        pdf_bytes = self.file.read()
        images = convert_from_path(BytesIO(pdf_bytes))
        ocr_text = []

        for i, image in enumerate(images):
            print(f"Processing EasyOCR for page {i + 1}/{len(images)}...")
            image_np = np.array(image)
            results = reader.readtext(image_np)

            page_text = ""
            for bbox, text, prob in results:
                page_text += text + "\n"
            ocr_text.append(page_text)

        return ocr_text

    def extract_metadata(self):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(self.file.read()))
            metadata = pdf_reader.metadata
            return metadata
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}
