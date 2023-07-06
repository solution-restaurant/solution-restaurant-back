from pydantic import BaseModel
class User(BaseModel):
    name: str
    pw: str
    message: str
    age: int