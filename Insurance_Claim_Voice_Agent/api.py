from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from logic import run
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017/")
collection = client["ai"]["insurance_data"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/start-agent")
async def start_agent(request: Request):
    body = await request.json()
    name = body.get("name")
    policy = body.get("policy")
    issue = body.get("issue")

    if not (name and policy and issue):
        return JSONResponse(status_code=400, content={"error": "Missing required fields."})

    patient_id = f"patient_{name.lower().replace(' ', '_')}_{policy}"

    patient_doc = {
        "_id": patient_id,
        "name": name,
        "policy": policy,
        "issue": issue,
        "status": "",
        "history": ""
    }

    collection.update_one({"_id": patient_id}, {"$set": patient_doc}, upsert=True)
    doc = collection.find_one({"_id": patient_id})

    run( doc)

    updated_doc = collection.find_one({"_id": patient_id})
    summary = updated_doc.get("summary", "No summary generated.")

    return JSONResponse(content = {"summary": summary})
