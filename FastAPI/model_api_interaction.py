from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class Prompt(BaseModel):
    text: str
    max_tokens: int = 200

@app.post("/generate")
def generate_response(prompt:Prompt):
    try:
        inputs = tokenizer(prompt.text, return_tensors = "pt")
        outputs = model.generate(inputs.input_ids, max_new_tokens = prompt.max_tokens)
        result = tokenizer.decode(outputs[0], skip_special_tokens = True)
        return {"Generated Text":result}

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))