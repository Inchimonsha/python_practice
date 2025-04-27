# import sys
# import os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uvicorn

from task_2_2.models.user import User
from task_2_2.routes.route import app

@app.post("/user")
def ch_user(user: User):
    response = User(**user.model_dump())
    return response

if __name__ == "__main__":
    uvicorn.run("task_2_2.app.main:app", host='127.0.0.1', port=8066, reload=True, workers=3)