from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from torch_utils import translate #Import predicting func

app = FastAPI(debug = True)

origins = ["*"]

app.add_middleware(  # Enable CORS
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class body_request(BaseModel):
    encSeq: str


@app.get("/")
async def home():
    return {"text" : "Hello word"}

@app.post("/predict")
async def predict(item: body_request):
    # Make prediction
    pred = translate(item.encSeq)
    return {"code" : 0, "result" : pred}

if __name__ == '__main__':
    uvicorn.run(app)
