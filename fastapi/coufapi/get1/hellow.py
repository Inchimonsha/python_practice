from fastapi import FastAPI
from fastapi.responses import FileResponse
# uvicorn coufapi.get1.hellow:my_awesome_api --reload --port 8081
my_awesome_api = FastAPI()

@my_awesome_api.get("/")
async def root():
    return FileResponse("coufapi/get1/index.html")