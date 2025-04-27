from fastapi import FastAPI
from pydantic import BaseModel

class Feedback(BaseModel):
    name: str
    message: str

app = FastAPI()



# Пример пользовательских данных (для демонстрационных целей)
fake_users = {
    1: {"username": "john_doe", "email": "john@example.com"},
    2: {"username": "jane_smith", "email": "jane@example.com"},
}

fake_comments = [
    {"name": "john_doe", "message": "john@example.com"},
    {"name": "jane_smith", "message": "jane@example.com"},
]

# Конечная точка для получения информации о пользователе по ID
@app.get("/users/{user_id}")
def read_user(user_id: int):
    return fake_users.get(user_id, {"error": "User not found"})

@app.get("/comments")
def get_all_comment():
    return fake_comments

@app.get("/comments/{user_id}")
def get_comment(user_id: int):
    return fake_comments[user_id]

@app.post("/feedback")
def feedback(fb: Feedback):
    fake_comments.append({"name": fb.name, "comments": fb.message})
    return f"Feedback received. Thank you, {fb.name}!"


# uvicorn end_point.main:app --reload --port 
# проверка
# import requests
# url = "http://127.0.0.1:8000/users/1"
# response = requests.get(url)
# print(response.json())