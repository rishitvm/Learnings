from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["ai"]
collection = db["insurance_data"]

def get_all_patients():
    return list(collection.find({"_id": {"$regex": "^patient_"}}))

def add_history(id, turn):
    collection.update_one({"_id": id}, {"$push": {"history": turn}}, upsert = True)

def get_context(id):
    doc = collection.find_one({"_id": id})
    if doc:
        return doc