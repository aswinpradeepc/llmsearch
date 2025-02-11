import os
import json
import fitz  # PyMuPDF
import pdfplumber

# Path to the documents folder
pdf_folder_path = '/media/documents'
output_folder_path = '/media/processed_data'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    document = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")  # Extract text
    return text

def extract_tables_from_pdf(pdf_path):
    """Extracts all tables from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            table = page.extract_tables()
            if table:
                tables.append(table)
    return tables

def process_pdf(pdf_path):
    """Extract text and tables from a single PDF file and save the results."""
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Extract tables
    tables = extract_tables_from_pdf(pdf_path)
    
    # Prepare the output data
    output_data = {
        "filename": os.path.basename(pdf_path),
        "text": text,
        "tables": tables,
    }
    
    # Save the extracted data to a JSON file
    output_file = os.path.join(output_folder_path, os.path.basename(pdf_path).replace(".pdf", ".json"))
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

def process_all_pdfs(pdf_folder_path):
    """Process all PDFs in the given folder."""
    for pdf_file in os.listdir(pdf_folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, pdf_file)
            print(f"Processing {pdf_file}...")
            process_pdf(pdf_path)
    print("All PDFs have been processed.")

# Start processing all PDFs in the folder
process_all_pdfs(pdf_folder_path)
