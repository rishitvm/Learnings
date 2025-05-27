from pydantic import BaseModel, Field, EmailStr, AnyUrl, field_validator, model_validator, computed_field
from typing import List, Dict, Optional, Annotated

class Patient(BaseModel):
    name: Annotated[str, Field(max_length = 50, title = "Patieent Name", description = "Null", examples = [])]
    email: EmailStr
    linkedin_url: AnyUrl
    age: Annotated[int, Field(gt = 0)]
    weight: Annotated[float, Field(gt = 0, strict = True)]
    height: Annotated[float, Field(gt = 0, strict = True)]
    married: Annotated[bool, Field(default = False)]
    allergies: Annotated[Optional[List[str]], Field(default = None, max_length = 5)]
    contact_details: Annotated[Dict[str,str], Field(max_length = 2)]

    @field_validator("email")
    @classmethod
    def email_validator(cls,value):
        domains = ["hdfc.com","icici.com"]
        my_value = value.split("@")[-1]
        if my_value not in domains:
            raise ValueError("Not in Domain List")
        return value
    
    @field_validator("name")
    @classmethod
    def name_validator(cls,value):
        return value.upper()
    
    @model_validator(mode = "after")
    def check(cls,model):
        if model.age > 60 and "emergency" not in model.contact_details:
            raise ValueError("Must have Emergency Contact")
        return model    
    
    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round((self.weight / (self.height ** 2)),2)
        return bmi

def insert(patient: Patient):
    print(patient.name)
    print(patient.email)
    print(patient.linkedin_url)
    print(patient.age)
    print(patient.weight)
    print(patient.married)
    print(patient.allergies)
    print(patient.contact_details)
    print(patient.bmi)

patient1_dict = {
    "name": "Person1",
    "email": "person@hdfc.com",
    "linkedin_url": "https://linkedin.com/in/person",
    "age": 65,
    "weight": 70.5,
    "height": 6,
    "contact_details": {"emergency":"1234098765", "Phone": "1234567890"}
}
patient1 = Patient(**patient1_dict)
insert(patient1)
