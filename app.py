
import torch
import numpy as np
from pdf2image import convert_from_path
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from doctr.models import ocr_predictor

import concurrent.futures

class PDFTranslator:
    def __init__(self, source_lang="en", target_lang="fr"):
        self.translator = Translator()
        self.model = ocr_predictor(pretrained=True).to("cuda")
        print(next(self.model.det_predictor.parameters()).device)  
        print(next(self.model.reco_predictor.parameters()).device)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model.eval()
        
        # Load multiple fonts for different languages
        self.fonts = {
            'en': 'arial.ttf',
            'fr': 'DejaVuSans.ttf',
            'hi': 'NotoSansDevanagari-Regular.ttf',
            'zh': 'NotoSansSC-Regular.ttf'
        }

    def detect_text(self, image_path):
        """Detect text using doctr with proper image loading"""
        print(f"Detecting text in image: {image_path}")

        # Load image and convert to RGB format
        image = Image.open(image_path).convert("RGB")  
        image_np = np.array(image)  # Convert the image to a NumPy array

        # Pass image as a **list** of numpy arrays to doctr
        results = self.model([image_np])  # Ensure it's a list

        text_data = []
        for element in results.pages[0].blocks:  # Adjusted for doctr output format
            for line in element.lines:
                text = " ".join([word.value for word in line.words])
                x_min, y_min = line.geometry[0]
                x_max, y_max = line.geometry[1]
                text_data.append({
                    "text": text,
                    "x": int(x_min * image.width),
                    "y": int(y_min * image.height),
                    "w": int((x_max - x_min) * image.width),
                    "h": int((y_max - y_min) * image.height)
                })

        print(f"Detected {len(text_data)} text elements.")
        return text_data

    def get_font(self, font_size):
        try:
            return ImageFont.truetype(self.fonts.get(self.target_lang, 'arial.ttf'), font_size)
        except:
            return ImageFont.load_default()

    def translate_with_retry(self, text, max_retries=3):
        """Translate text with retry mechanism"""
        for attempt in range(max_retries):
            try:
                if not text.strip():
                    return text
                translated = self.translator.translate(text, src=self.source_lang, dest=self.target_lang)
                return translated.text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Translation failed after {max_retries} attempts for: {text}")
                    return text
                continue

    def preserve_formatting(self, image, text_info, translated_text):
        """Preserve text formatting while replacing with translation"""
        draw = ImageDraw.Draw(image)
        x, y, w, h = text_info['x'], text_info['y'], text_info['w'], text_info['h']  # Fixed bbox access

        # Sample multiple points for background color
        bg_colors = []
        for i in range(3):
            for j in range(3):
                sample_x = x + (w * i // 2)
                sample_y = y + (h * j // 2)
                try:
                    bg_colors.append(image.getpixel((sample_x, sample_y)))
                except:
                    continue

        # Use most common background color
        bg_color = max(set(bg_colors), key=bg_colors.count) if bg_colors else (255, 255, 255)

        # Calculate appropriate font size
        font_size = int(h * 0.8)  # Estimate font size from height
        font = self.get_font(font_size)

        # Measure text and adjust placement
        text_bbox = draw.textbbox((0, 0), translated_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Scale text if necessary
        if text_width > w:
            font_size = int(font_size * (w / text_width) * 0.95)
            font = self.get_font(font_size)
            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

        # Clear background
        draw.rectangle([x, y, x + w, y + h], fill=bg_color)

        # Center text
        text_x = x + (w - text_width) / 2
        text_y = y + (h - text_height) / 2

        # Draw text with contrast-based color
        text_color = (0, 0, 0) if sum(bg_color[:3]) > 382 else (255, 255, 255)
        draw.text((text_x, text_y), translated_text, font=font, fill=text_color)

        return image

    def process_page(self, image, page_number):
        img_path = f"temp_page_{page_number}.png"
        image.save(img_path)

        # Detect text
        text_data = self.detect_text(img_path)

        # Process image
        processed_image = Image.open(img_path)

        # Process detected text
        for text_elem in text_data:
            translated_text = self.translate_with_retry(text_elem['text'])
            processed_image = self.preserve_formatting(
                processed_image,
                text_elem,
                translated_text
            )

        # Save processed image
        processed_path = f"processed_page_{page_number}.png"
        processed_image.save(processed_path)
        os.remove(img_path)

        return page_number, processed_path

    def process_pdf(self, input_pdf, output_pdf):
        """Main processing function"""
        print(f"Processing PDF: {input_pdf}")

        # Convert PDF to images
        images = convert_from_path(input_pdf, dpi=300)
        processed_images = {}

        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.process_page, image, i) for i, image in enumerate(images)]
            for future in concurrent.futures.as_completed(futures):
                page_number, processed_path = future.result()
                processed_images[page_number] = processed_path

        # Sort processed images by page number
        sorted_processed_images = [processed_images[i] for i in sorted(processed_images.keys())]

        # Convert back to PDF
        self.images_to_pdf(sorted_processed_images, output_pdf)

        # Cleanup
        for img_path in sorted_processed_images:
            os.remove(img_path)

        print(f"Processing complete. Output saved to: {output_pdf}")

    def images_to_pdf(self, image_paths, output_pdf):
        """Convert images to PDF while preserving quality"""
        first_image = Image.open(image_paths[0])
        width, height = first_image.size

        c = canvas.Canvas(output_pdf, pagesize=(width, height))
        for img_path in image_paths:
            img = Image.open(img_path)
            c.drawImage(img_path, 0, 0, width, height)
            c.showPage()
        c.save()

# Usage
if __name__ == "__main__":
    translator = PDFTranslator(source_lang="en", target_lang="fr")
    translator.process_pdf("input.pdf", "translated_output.pdf")