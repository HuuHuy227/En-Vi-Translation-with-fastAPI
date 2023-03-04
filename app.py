from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import torch
from model.transformer import Seq2SeqTransformer
from utils.utils import Translation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMER_WEIGHTS = './weights/viEn_transformer.pth'
VOCAB_PATH = './weights/vocabs.pkl'
vocab_src_path = './weights/vocab_src.pkl'
vocab_tgt_path = './weights/vocab_tgt.pkl'

EMB_SIZE = 512
NHEAD = 8 # embed_dim must be divisible by num_heads
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROP_OUT = 0.1

def load_model():
    translation = Translation(vocab_src_path, vocab_tgt_path, DEVICE)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = translation.len_vocab()

    # Load transformer model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)

    transformer.load_state_dict(torch.load(TRANSFORMER_WEIGHTS,map_location = DEVICE)) 
    transformer = transformer.to(DEVICE)

    return transformer, translation  

app = FastAPI(debug = True)
transformer, translation = load_model()
origins = ["*"]

app.add_middleware(  # Enable CORS
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class body_request(BaseModel):
    seq: str

@app.get("/")
async def home():
    return {"text" : "Hello word"}

@app.post("/translate")
async def predict(item: body_request):
    # Make prediction
    translated = translation.translate(transformer, item.seq)
    return {"code" : 0, "result" : translated}

if __name__ == '__main__':
    uvicorn.run(app)