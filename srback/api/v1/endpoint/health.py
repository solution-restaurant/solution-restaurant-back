from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from srback.llm.lclogic import lcmain

import json
import urllib.request
from srback.core.config import settings
from pydantic import BaseModel

from srback.db.models import users
from srback.db.db import conn
from srback.db.user import User

router = APIRouter()

#uvicorn main:app --reload

class Model(BaseModel):
    content: str
    userName: str
    
class JoinModel(BaseModel):
    userName: str
    password: str

class LoginModel(BaseModel):
    userName: str
    password: str


@router.post("/join")
def user_join(data : JoinModel):
  print("receive message : " + data.userName +" from : " +data.password)
  # example = conn.execute(users.select()).fetchall()
  example = conn.execute(users.insert().values(name=data.userName,
                pw=data.password, message='first', age=20))
  print("example : " + str(example))
  return data

  
@router.post("/login")
def user_login(data : LoginModel):
  print("receive message : " + data.userName +" from : " +data.password)
  example = conn.execute(users.select().where(users.c.name == data.userName, users.c.pw == data.password)).first()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data


@router.post("/send")
def receive_message(data : Model):
  print("receive message : " + data.content +" from : " +data.userName)
  # example = conn.execute(users.select()).fetchall()
  data.userName='영양코칭AI'
  data.content =lcmain.health_agents(data.content)
  # data.content= lcmain.custom_health_agent(data.content)
  print(data)
  return data

# @router.post("/send")
# def receive_message(data : Model):
#   print("receive message : " + data.content +" from : " +data.userName)
#   # example = conn.execute(users.select()).fetchall()
#   example = conn.execute(users.select().where(users.c.message == data.content)).first()
#   print("example : " + str(example))
#   data.userName='영양코칭AI'
#   data.content= lcmain.custom_health_agent(data.content)
#   print(data)
#   return data

@router.get("/")
def read_root():
    return {"Hello world"}


