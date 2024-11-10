import json
import logging
import os

import aiohttp


logger = logging.getLogger("pdf_analyzer_app.api")
logging.basicConfig(level=logging.DEBUG)

class OllamaAPI:
    def __init__(self, model=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q8_0")
        self.base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    async def summarize_text(self, text):
        if len(text) > 2000:
            text = text[:2000]
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"Summarize the following text:\n{text}"
        }
        return await self._send_request(url, payload)

    async def answer_question(self, text, question):
        if len(text) > 5000:
            text = text[:5000]

        url = f"{self.base_url}/api/generate"
        prompt = (
            f"Text: {text}\n"
            f"Question: {question}\n"
            # f"If you find relevant data, respond with a clear positive statement, like 'I found the relevant information.' "
            # f"If you do not find any relevant data, respond with a clear negative statement, like 'I did not find any relevant information.' "
            # f"Ensure that your response reflects the sentiment accurately so it can be analyzed effectively."
        )

        payload = {
            "model": self.model,
            "prompt": prompt
        }

        return await self._send_request(url, payload)

    async def ask_llama(self, text, model_version):
        url = f"{self.base_url}/api/generate"
        prompt = f"""
        You are an advanced fraud detection system. Analyze the following text and metadata extracted from a PDF document for any signs of fraud. Highlight if the document might be fake, fraudulent, or contains suspicious information.

        Text:
        {text}
        """
        payload = {
            "model": model_version,
            "prompt": prompt
        }
        return await self._send_request(url, payload)

    async def ask_llama_summary(self, text):
        url = f"{self.base_url}/api/generate"


        prompt = f"""
        You are an advanced summarization system. Summarize the following information extracted from a PDF document. The summary should include the main points, important information, and any relevant references to the original text.

        Text:
        {text}
        """

        payload = {
            "model": "llama3:8b-instruct-q8_0",
            "prompt": prompt
        }

        return await self._send_request(url, payload)

    async def _send_request(self, url, payload):
        try:
            logger.debug("Sending request to %s with payload: %s", url, payload)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    complete_response = ""

                    async for line in response.content:
                        if line:
                            try:
                                line = line.decode("utf-8")
                                response_data = json.loads(line)
                                complete_response += response_data.get("response", "")
                                logger.debug("Partial response: %s", response_data)
                                if response_data.get("done", False):
                                    break
                            except ValueError as json_err:
                                logger.error("Failed to decode JSON: %s", json_err)
                                logger.error("Raw line: %s", line)
                                continue

                    logger.debug("Final response received: %s", complete_response)
                    return {
                        "answer": complete_response.strip()
                    }

        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
            return {"error": str(http_err)}
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            return {"error": "An unexpected error occurred."}




