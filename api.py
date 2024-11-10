import logging
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import JSONResponse

from pdf_analyzer_app import pdf_processor_2
from pdf_analyzer_app.ollama_api import OllamaAPI
from pdf_analyzer_app.pdf_analyzer import PDFAnalyzer
from pdf_analyzer_app.pdf_processor import PDFProcessor
from pdf_analyzer_app.pdf_processor_2 import PDFProcessor2
from pdf_analyzer_app.sentiment_analysis import SentimentAnalyzer
from pdf_analyzer_app.fraud_detector import FraudDetector
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, Query
from pdf_analyzer_app.nvidia_api import NvidiaAPI

from fastapi import HTTPException

MAX_CONCURRENT_TASKS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
ollama_api = OllamaAPI()
sentiment_analyzer = SentimentAnalyzer()
fraud_detector = FraudDetector()

def create_pdf(content, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 12)
    margin = 50
    text_object = c.beginText(margin, height - margin)

    lines = content.split('\n')
    current_height = height - margin

    for line in lines:
        words = line.split(' ')
        current_line = ''
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_width = c.stringWidth(test_line, "Helvetica", 12)
            if text_width < (width - 2 * margin):
                current_line = test_line
            else:
                text_object.textLine(current_line)
                current_line = word
                current_height -= 14
                if current_height < margin:
                    c.showPage()
                    current_height = height - margin
        if current_line:
            text_object.textLine(current_line)
    c.drawText(text_object)
    c.save()

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info("Received a request to upload a PDF file.")
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor(file.file)
    logger.info("Processing PDF file...")
    analyzer = PDFAnalyzer(pdf_processor, ollama_api)
    summary = analyzer.analyze_pdf()
    logger.info("PDF processing complete.")
    return {"summary": summary}


@app.post("/ask_question/")
async def ask_question(question: str = Form(...), file: UploadFile = File(...)):
    logger.info("Received question: %s", question)
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor(file.file)
    logger.info("Processing PDF file for question...")

    try:
        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text from pages.")


        text = "\n".join(extracted_pages)

        logger.debug("Combined extracted PDF text: %s", text[:500])
        if not text.strip():
            return {"error": "No text found in the PDF to process."}
    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return {"error": "Error extracting text from PDF."}

    try:

        answer = await ollama_api.answer_question(text, question)
    except Exception as e:
        logger.error("Error getting answer from API: %s", e)
        return {"error": "Error getting answer from the question."}

    logger.info("Question processing complete.")
    return {"answer": answer}


@app.post("/generate_pdf/")
async def generate_pdf(file: UploadFile = File(...), instruction: str = Form(...)):
    logger.info("Received request to generate a PDF.")

    pdf_processor = PDFProcessor(file.file)
    logger.info("Extracting text from PDF...")

    try:

        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text from pages.")

        full_text = "\n".join(extracted_pages)
    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return {"error": "Error extracting text from PDF."}

    try:
        generated_content = await ollama_api.answer_question(full_text, instruction)
    except Exception as e:
        logger.error("Error getting generated content from API: %s", e)
        return {"error": "Error generating summary from the extracted text."}

    logger.info("Generated content: %s", generated_content)

    pdf_filename = "generated_output.pdf"
    create_pdf(generated_content['answer'], pdf_filename)
    logger.info("PDF generated successfully.")

    return {"message": "PDF generated successfully", "filename": pdf_filename}


@app.post("/analyze_sentiment/")
async def analyze_sentiment(file: UploadFile = File(...)):
    logger.info("Received request to analyze sentiment from PDF.")

    pdf_processor = PDFProcessor(file.file)
    logger.info("Extracting text from PDF for sentiment analysis...")
    try:
        text = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text for sentiment analysis: %s", text)
    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return {"error": "Error extracting text from PDF."}

    sentiment_result = sentiment_analyzer.analyze(text)
    logger.info("Sentiment analysis complete.")
    return {"sentiment": sentiment_result}

@app.post("/detect_fraud/")
async def detect_fraud(file: UploadFile = File(...)):
    logger.info("Received a request to detect fraud in a PDF file.")
    pdf_processor = PDFProcessor(file.file)
    logger.info("Extracting text and metadata from PDF...")

    try:
        text = pdf_processor.extract_text_by_page()
        metadata = pdf_processor.extract_metadata()
        logger.debug("Extracted PDF text and metadata.")
    except Exception as e:
        logger.error("Error extracting text or metadata from PDF: %s", e)
        return {"error": "Error extracting text or metadata from PDF."}

    is_fraudulent = fraud_detector.detect_fraud(text, metadata)
    logger.info("Fraud detection complete.")
    return {"is_fraudulent": is_fraudulent}


@app.post("/detect_fraud_llama/")
async def detect_fraud_llama(file: UploadFile = File(...), model_version: str = Form("llama3.2")):
    logger.info("Received a request to detect fraud in a PDF using Llama model.")
    logger.debug(f"Model version: {model_version}")

    pdf_processor = PDFProcessor(file.file)
    try:
        logger.info("Extracting text and metadata from PDF...")
        extracted_pages = pdf_processor.extract_text_by_page()
        metadata = pdf_processor.extract_metadata()
        logger.debug(f"Extracted text from pages: {[text[:500] for text in extracted_pages]}...")
    except Exception as e:
        logger.error(f"Error extracting text or metadata from PDF: {e}")
        return {"error": "Error extracting text or metadata from PDF."}


    combined_text = "\n".join(extracted_pages)

    logger.info("Sending data to Llama model for fraud detection...")
    prompt = f"""
    You are an advanced fraud detection system. Analyze the following text and metadata extracted from a PDF document for any signs of fraud. Provide a percentage probability that the document is fraudulent, along with reasons for your assessment.

    Text:
    {combined_text}

    Metadata:
    {metadata}

    Respond with a percentage and reasons why the document may be considered fraudulent.
    """

    response = await ollama_api.ask_llama(prompt, model_version=model_version)

    logger.info(f"Llama model response: {response}")
    return {"fraud_detection_result": response}


@app.post("/ask_question_2/")
async def ask_question_2(question: str = Form(...), file: UploadFile = File(...)):
    logger.info("Received question: %s", question)
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor(file.file)
    logger.info("Processing PDF file for question...")

    try:
        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text for pages.")

        page_references = []
        all_answers = []

        for page_number, text in enumerate(extracted_pages, start=1):
            logger.debug("Processing page %d...", page_number)


            chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]

            for chunk in chunks:
                answer = await ollama_api.answer_question(chunk, question)
                logger.debug(f"Answer from Ollama for page {page_number}: {answer}")

                if is_relevant_answer(answer['answer']):
                    logger.info(f"Relevant answer found on page {page_number}")
                    page_references.append(page_number)
                    all_answers.append(answer['answer'])
                    break

        if page_references:
            logger.info(f"Information found on pages: {page_references}")
            return {
                "answer": all_answers,
                "page_references": page_references
            }
        else:
            return {"answer": "No relevant information found.", "page_references": []}

    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return {"error": "Error extracting text from PDF."}

import asyncio


@app.post("/ask_question_3/")
async def ask_question_3(question: str = Form(...), file: UploadFile = File(...)):
    logger.info("Received question: %s", question)
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor(file.file)
    logger.info("Processing PDF file for question...")

    try:
        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text for pages.")

        page_references = []
        all_answers = []

        semaphore = asyncio.Semaphore(25)

        async def process_chunk(chunk, page_number):
            async with semaphore:
                answer = await ollama_api.answer_question(chunk, question)
                if is_relevant_answer(answer['answer']):
                    logger.info(f"Relevant answer found on page {page_number}")
                    return (page_number, answer['answer'])
                return None

        tasks = []
        for page_number, text in enumerate(extracted_pages, start=1):
            logger.debug("Processing page %d...", page_number)

            chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]

            for chunk in chunks:
                tasks.append(process_chunk(chunk, page_number))

        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                page_ref, answer = result
                page_references.append(page_ref)
                all_answers.append(answer)

        if page_references:
            logger.info(f"Information found on pages: {page_references}")

            summary_input = f"""
            Question: {question}

            Answers:
            {" ".join(all_answers)}

            Pages referenced: {page_references}
            """
            logger.info("Sending answers and references to Llama for summary generation...")

            summary_response = await ollama_api.ask_llama_summary(summary_input)

            logger.info(f"Summary response from Llama: {summary_response}")
            return {
                "summary": summary_response['answer']
            }
        else:
            return {"answer": "No relevant information found.", "page_references": [], "summary": None}

    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return {"error": "Error extracting text from PDF."}


from fastapi import HTTPException

@app.post("/count_tokens/")
async def count_tokens(file: UploadFile = File(...)):
    logger.info("Received request to count tokens in PDF.")
    pdf_processor = PDFProcessor(file.file)

    try:

        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text from pages.")

        tokens_per_page = []
        total_tokens = 0

        for page_number, text in enumerate(extracted_pages, start=1):
            tokens = text.split()
            num_tokens = len(tokens)
            tokens_per_page.append(num_tokens)
            total_tokens += num_tokens

            logger.debug(f"Page {page_number}: {num_tokens} tokens")

        logger.info("Token counting complete.")
        return {
            "tokens_per_page": tokens_per_page,
            "total_tokens": total_tokens
        }
    except Exception as e:
        logger.error("Error counting tokens in PDF: %s", e)
        raise HTTPException(status_code=500, detail="Error counting tokens in PDF.")

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_results_db"]
collection = db["results_collection2"]


import spacy

nlp = spacy.load("en_core_web_md")

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def calculate_satisfaction(answer, question):
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(question_embedding, answer_embedding).item()

    if any(term in answer.lower() for term in ["error", "not related", "unrelated", "mistake", "irrelevant"]):
        return 0

    return similarity * 100


nvidia_api = NvidiaAPI(api_key="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")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate_text/")
async def generate_text(request: PromptRequest):
    result = await nvidia_api.generate_text(request.prompt)
    return result


@app.post("/upload_pdf_token/")
async def upload_pdf_token(file: UploadFile = File(...)):
    logger.info("Received a request to upload a PDF file.")
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor2(file.file)

    logger.info("Extracting text from PDF...")
    extracted_text = pdf_processor.extract_text_by_page()

    combined_summary = {
        "text": "\n".join(extracted_text),
    }

    logger.info("PDF processing complete.")
    return JSONResponse(content={"summary": combined_summary})


@app.post("/ask_question_par/")
async def ask_question_par(question: str = Form(...), file: UploadFile = File(...)):
    logger.info("Received question: %s", question)
    logger.debug("File details: %s", file.filename)

    pdf_processor = PDFProcessor(file.file)
    logger.info("Processing PDF file for question...")

    try:
        extracted_pages = pdf_processor.extract_text_by_page()
        logger.debug("Extracted PDF text for pages.")

        semaphore = asyncio.Semaphore(5)

        async def process_chunk(chunk, page_number):
            async with semaphore:
                answer = await ollama_api.answer_question(chunk, question)
                logger.info(f"Answer found for pages {page_number}: {answer['answer']}")

                if answer and 'answer' in answer:
                    page_pair_tuple = (page_number[0], page_number[1])
                    pair_result = {
                        "question": question,
                        "pages": list(page_pair_tuple),
                        "answer": answer['answer']
                    }

                    logger.debug(f"Pair result before insertion: {pair_result}")

                    try:
                        collection.insert_one(pair_result)
                        logger.info(f"Inserted result for pages {page_pair_tuple} into MongoDB.")
                    except Exception as db_error:
                        logger.error(f"Failed to insert result for pages {page_pair_tuple} into MongoDB: {db_error}")

                    return (page_pair_tuple, answer['answer'])
                else:
                    logger.error(f"No valid answer returned for pages {page_number}: {answer}")

        tasks = []

        for i in range(0, len(extracted_pages), 2):
            page_number_1 = i + 1
            page_number_2 = i + 2 if i + 1 < len(extracted_pages) else None

            combined_text = extracted_pages[i]
            if page_number_2 is not None:
                combined_text += extracted_pages[i + 1]

            logger.debug(f"Processing page pair {page_number_1} and {page_number_2 if page_number_2 else 'N/A'}")

            chunks = [combined_text[j:j + 5000] for j in range(0, len(combined_text), 5000)]

            for chunk in chunks:
                tasks.append(process_chunk(chunk, (page_number_1, page_number_2)))

        results = await asyncio.gather(*tasks)

        satisfaction_results = []
        total_percentage = 0
        count = 0

        for result in results:
            if result:
                pages, answer = result
                satisfaction_percentage = calculate_satisfaction(answer, question)
                satisfaction_results.append({"pages": list(pages), "result": satisfaction_percentage})

                total_percentage += satisfaction_percentage
                count += 1

        general_percentage = total_percentage / count if count > 0 else 0

        return {
            "status": "Processing complete",
            "satisfaction_results": satisfaction_results,
            "general_result": general_percentage
        }

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": "Error processing PDF."}

def is_relevant_answer(answer: str) -> bool:
    sentiment_analyzer = SentimentAnalyzer()
    sentiment = sentiment_analyzer.analyze(answer)

    logger.debug(f"Sentiment analysis for answer: {sentiment}")


    positive_phrases = [
        "I found the relevant information",
        "I have the information you requested",
        "Here is the information you were looking for",
    ]

    if any(phrase in answer for phrase in positive_phrases):
        logger.info("Answer is considered relevant due to explicit positive statement.")
        return True

    if sentiment['positive'] > 50:
        logger.info(f"Answer is considered relevant due to positive sentiment: {sentiment['description']}")
        return True

    elif sentiment['negative'] > 50:
        logger.info(f"Answer is considered not relevant due to negative sentiment: {sentiment['description']}")
        return True

    elif sentiment['neutral'] > 50:
        logger.info(f"Answer is neutral, checking for keywords.")
        return True

    keywords = ["offer", "discount", "promotion", "deal", "expires", "valid until"]

    if len(answer.strip()) < 20:
        return False

    for keyword in keywords:
        if keyword.lower() in answer.lower():
            return True

    generic_responses = [
        "I'm sorry, I don't know",
        "No relevant information",
        "I cannot find anything",
        "The text doesn't mention",
    ]

    for generic in generic_responses:
        if generic.lower() in answer.lower():
            return False

    return False






