import uvicorn as uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/", response_class=FileResponse)
def root_html():
    return "coufapi/hard_postcalc/index.html"


@app.post("/calculate")
def calculate(num1: int = Form(ge=0, lt=111), num2: int = Form(ge=0, lt=111)):
    print("num1 =", num1, "   num2 =", num2)
    return {"result": num1 + num2}


@app.get("/calculate", response_class=FileResponse)
def calc_form():
    return "coufapi/hard_postcalc/calculate.html"


if __name__ == '__main__':
    uvicorn.run(app,
                host='127.0.0.1',
                port=8082)