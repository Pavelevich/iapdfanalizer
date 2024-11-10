class PDFAnalyzer:
    def __init__(self, pdf_processor, ollama_api):
        self.pdf_processor = pdf_processor
        self.ollama_api = ollama_api
        self.summary = ""

    def analyze_pdf(self):

        text = self.pdf_processor.extract_text()
        self.summary = self.ollama_api.summarize_text(text)
        return self.summary

    def ask_question(self, question):

        text = self.pdf_processor.text
        return self.ollama_api.answer_question(text, question)
