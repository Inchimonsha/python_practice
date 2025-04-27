import uvicorn
from fastapi import FastAPI, Body, HTTPException
from pydantic import ValidationError
from models.User import User

title = 'My first App'

app = FastAPI(title=title)

users = []

user = User(id=1, name="John Doe")
users.append(user)

# пример роута (маршрута)
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/users")
def read_users():
    return users

@app.get("/users", response_model=User)
def user_root():
    return user

# from fastapi.responses import JSONResponse
# @app.get("/")
# async def read_user() -> JSONResponse:
#     return JSONResponse(content=user.dict())

@app.post("/custom")
def read_custom_message(user: User):
    return {"message": f"This is a custom message: {user.name}!"}

@app.post("/add")
def add_custom_message(user: User):
    users.append(user)
    return user

if __name__ == "__main__":
    # fastapi [dev|run] main.py
    uvicorn.run("base_struct.app.main:app", host='127.0.0.1', port=8066, reload=True, workers=3)