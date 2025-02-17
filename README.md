# TransDocAI

TransDocAI is an AI-powered tool for extracting and translating text from PDFs while preserving formatting. It leverages OCR (Optical Character Recognition) using **DocTR**, along with Google Translate, to provide accurate translations.

## Features
- Extracts text from PDFs using **Doctr OCR**
- Translates text to multiple languages using **Google Translate**
- Preserves the formatting and positioning of text within the document
- Processes PDFs in parallel for faster execution
- Converts processed images back into a PDF format

## Installation

Ensure you have Python installed (>=3.9) and then install the dependencies:

```sh
pip install -r requirements.txt
```

You may also need to install additional system dependencies for PDF and OCR processing:

```sh
sudo apt-get install poppler-utils 
```

## Usage

To translate a PDF file from English to French, run:

```sh
python main.py --input input.pdf --output translated_output.pdf --source en --target fr
```

### Arguments
- `--input`: Path to the input PDF file
- `--output`: Path to save the translated PDF
- `--source`: Source language (default: English - `en`)
- `--target`: Target language (default: French - `fr`)

## GPU Acceleration
To ensure that the model is running on GPU, verify using:

```python
import torch
print(torch.cuda.is_available())
```

For **DocTR** to utilize the GPU, modify the OCR model initialization:

```python
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True).to('cuda')
```

## Dependencies
- **doctr** – OCR engine for text detection
- **pdf2image** – Converts PDFs to images
- **Pillow** – Image processing
- **Googletrans** – Text translation
- **ReportLab** – PDF generation

## License
This project is open-source under the **Apache License 2.0**.

