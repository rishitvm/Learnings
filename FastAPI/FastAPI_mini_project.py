from fastapi import FastAPI, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Literal, Annotated
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
import json

app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str, Field(..., description = "Patient ID")]
    name: Annotated[str, Field(..., description = "Patient Name", max_length = 50)]
    city: Annotated[str, Field(..., description = "Patient City", max_length = 50)]
    age: Annotated[int, Field(..., description = "Patient Age", gt = 0)]
    gender: Annotated[Literal["male","female","others"], Field(..., description = "Patient Gender")]
    height: Annotated[float, Field(..., description = "Patient Height", gt = 0)]
    weight: Annotated[float, Field(..., description = "Patient Weight", gt = 0)]

    @computed_field
    @property
    def bmi(self) -> float:
        calculated_bmi = round((self.weight / (self.height)**2),2)
        return calculated_bmi
    
    @computed_field
    @property
    def verdict(self) -> str:
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Normal"
        else:
            return "Overweight"
        
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0)]
    gender: Annotated[Optional[Literal['male', 'female']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None, gt=0)]
    weight: Annotated[Optional[float], Field(default=None, gt=0)]

def load_data():
    with open("patients.json","r") as f:
        data = json.load(f)
    return data

def save_data(data):
    with open("patients.json","w") as f:
        json.dump(data,f)

@app.get("/")
def hello_page():
    return {"message":"This is Home Page"}

@app.get("/about")
def about_page():
    return {"message":"This is a Mini Project for practice"}

@app.get("/view")
def view_patients():
    data = load_data()
    return data

@app.get("/patient/{patient_id}")
def get_patient(patient_id: str = Path(...,description = "Provide Patient ID")):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code = 404, detail = "Patient not found")
    patient_data = data[patient_id]
    return patient_data

@app.get("/sort")
def sort_data(sort_param: str = Query(...), order_by: str = Query('asc')):
    valid_fields = ["height","weight","bmi"]
    if sort_param not in valid_fields:
        raise HTTPException(status_code = 400, detail = "Invalid")
    if order_by not in ["asc","desc"]:
        raise HTTPException(status_code = 400, detail = "Invalid Request")
    data = load_data()
    temp = False if order_by == "asc" else True
    sorted_data = sorted(data.values(), key = lambda x:x.get(sort_param,0), reverse = temp)
    return sorted_data

@app.post("/create")
def create_patient(patient: Patient):
    data = load_data()
    if patient.id in data:
        raise HTTPException(status_code = 404, detail = "Patient Already exists")
    data[patient.id] = patient.model_dump(exclude = ["id"])
    save_data(data)
    return JSONResponse(status_code = 201, content = {"message":"Successfully Created"})

@app.delete("/delete/{patient_id}")
def delete_patient(patient_id: str = Path(...)):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code = 404, detail = "Patient not found")
    del data[patient_id]
    save_data(data)
    return JSONResponse(status_code = 201, content = {"message":"Record Deleted Successfully"})

@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update: PatientUpdate):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    existing_patient_info = data[patient_id]
    updated_patient_info = patient_update.model_dump(exclude_unset=True)
    for key, value in updated_patient_info.items():
        existing_patient_info[key] = value
    existing_patient_info['id'] = patient_id
    patient_pydandic_obj = Patient(**existing_patient_info)
    existing_patient_info = patient_pydandic_obj.model_dump(exclude='id')
    data[patient_id] = existing_patient_info
    save_data(data)
    return JSONResponse(status_code=200, content={'message':'patient updated'})