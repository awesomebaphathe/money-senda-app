import io
import logging
from typing import Optional

import numpy as np
import uvicorn as uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from OCRImages import OCRImage
from chatGPT import ChatGPT

app = FastAPI()

# Instantiate chatbot models
chatbot = ChatGPT()

class InputTextModel(BaseModel):
    text: str

class InputImageModel(BaseModel):
    image: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class OutputModel(BaseModel):
    response: Optional[str] = None
    image: Optional[bytes] = None

# Configure logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.post("/completion", response_model=OutputModel)
async def get_chatbot_response(request: Request, text: Optional[str] = None, image: Optional[UploadFile] = File(None)):
    client_ip = request.client.host

    if not text and not image:
        raise HTTPException(status_code=400, detail="Please provide either text or an image.")

    # If text input is provided, use it as the input to the chatbot
    if text:
        user_input = text.strip()
    else:
        # If only an image is provided, use it for OCR processing and chatbot input
        ocr = OCRImage()
        try:
            image_data = io.BytesIO(image.file.read())
            image_text = ocr.process_image(image_data)
            print(image_text)
            if image_text.isspace():
                raise HTTPException(status_code=400, detail="Could not extract text from image. Please try again.")
            user_input = image_text.strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Error processing image. Please try again.")

    try:
        response = chatbot.get_completion(user_input)

        # Log the input and response
        logging.info(f"IP: {client_ip} - Input: {user_input} - Output: {response}")

        # If the response contains a path to an image, return the image as the response
        if response.startswith("path:"):
            with open(response[5:], "rb") as f:
                image_bytes = f.read()
            return {"image": image_bytes}

        # Otherwise, return the response as text
        return {"response": response}

    except Exception as e:
        logging.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
