from fastapi import FastAPI
from pydantic import BaseModel, validator
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
import uvicorn
from deepdoctection.ocr_detector import OCRDetector

app = FastAPI(title="Table Segmentation",description="API for table segmentation and ocr in base64 image",version="1.0")

model = OCRDetector()

class ImageBase64(BaseModel):
    image: bytes
    
    @validator("image")
    def type_checker(cls,image):
        if not isinstance(image,bytes):
            raise TypeError("Input bytes")
        
        elif len(image) ==0:
            raise ValueError("Image length should be more than zero")
        
        return image

@app.post('/api/table_segmentation')
async def table_segmentation(image_base64: ImageBase64):
    try:
        image_np=np.array(Image.open(BytesIO(base64.b64decode(image_base64.image)))) #read immage
        
    except UnidentifiedImageError as ex:
        return JSONResponse(status_code=400, content={"UnidentifiedImageError":str(ex)})
    
    except Exception as ex:
        return JSONResponse(status_code=500, content={"Other Error":str(ex)})

    try:
        _, dfs = model.predict(image_np)
        response = {
            "image_length": len(dfs),
            "dataframes": [df.to_json() for df in dfs]
        }
        return JSONResponse(status_code=200, content=response)
    
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error":str(ex)})


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")    