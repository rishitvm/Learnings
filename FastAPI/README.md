# FastAPI

This repository contains a series of Basics & Projects implementing AI models and interacting with FastAPI.

---

## 📁 Contents

### `Pydantic_basics.py`

This script demonstrates Pydantic's advanced validation and schema definition features using a realistic patient data model. It's designed for learning and practicing how to enforce rules, compute derived values, and build safe schemas.

---

### `FastAPI_mini_project.py`

This is a mini FastAPI project that manages patient health records using a JSON file as the database. It demonstrates RESTful API design, input validation using Pydantic, computed fields like BMI, and sorting/filtering functionality.

### Endpoints Created

| Method | Route                   | Description                           |
|--------|-------------------------|---------------------------------------|
| GET    | `/`                     | Home route                            |
| GET    | `/about`                | About the project                     |
| GET    | `/view`                 | View all patients                     |
| GET    | `/patient/{id}`         | Get specific patient by ID            |
| GET    | `/sort`                 | Sort patients by height/weight/bmi    |
| POST   | `/create`               | Add a new patient                     |
| PUT    | `/edit/{id}`            | Update existing patient info          |
| DELETE | `/delete/{id}`          | Delete a patient                      |

---

