# from fastapi import FastAPI
# from pydantic import BaseModel
# from pdf_analyzer_app.nvidia_api import NvidiaAPI
#
# app = FastAPI()
# nvidia_api = NvidiaAPI(api_key="")
#
# class PromptRequest(BaseModel):
#     prompt: str
#
# @app.post("/generate_text/")
# async def generate_text(request: PromptRequest):
#     result = await nvidia_api.generate_text(request.prompt)
#     return result
