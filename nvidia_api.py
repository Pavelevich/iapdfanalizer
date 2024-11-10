import aiohttp
import json
import logging

logger = logging.getLogger("pdf_analyzer_app.api")
logging.basicConfig(level=logging.DEBUG)

class NvidiaAPI:
    def __init__(self, api_key, model="nvidia/llama-3.1-nemotron-70b-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    async def generate_text(self, prompt, temperature=0.5, top_p=1, max_tokens=1024):
        url = self.base_url
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        return await self._send_request(url, payload, headers)

    async def _send_request(self, url, payload, headers):
        try:
            logger.debug("Sending request to %s with payload: %s", url, payload)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    logger.debug("Response data: %s", response_data)
                    return response_data
        except aiohttp.ClientResponseError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
            return {"error": str(http_err)}
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            return {"error": "An unexpected error occurred."}
