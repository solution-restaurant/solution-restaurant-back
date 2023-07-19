from typing import Any, List, Union
from fastapi import APIRouter, Depends, HTTPException
from srback.llm.lclogic import lcmain

import json
import urllib.request
from srback.core.config import settings
from pydantic import BaseModel

from srback.db.models import users
from srback.db.models import alarms
from srback.db.db import conn

from srback.db.db import SessionLocal
from sqlalchemy import update

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
    allergy: str
    disease: str
    alarmTime: str
    alarmState: str

class AlarmModel(BaseModel):
    userMealId: str
    userName: str
    userRecoMeal: str
    userResOfAi: str
    userGoodFB: str
    userBadFB: str
    userChkEat: str
    userMLD: str
    userAlarmImg: str
    userMealInfo: Union[List[str], None] = None

@router.post("/join")
def user_join(data : JoinModel):
  print("receive message : " + data.userName +" from : " +data.password)
  # example = conn.execute(users.select()).fetchall()
  example = conn.execute(users.insert().values(name=data.userName,
                pw=data.password, message='first', age=20))
  conn.commit()
  print("example : " + str(example))
  return data

  
@router.post("/login")
def user_login(data : LoginModel):
  print("receive message : " + data.userName +" from : " +data.password)
  example = conn.execute(users.select().where(users.c.name == data.userName, users.c.pw == data.password)).first()
  conn.commit()
  data.allergy = example.allergy
  data.disease = example.disease
  data.alarmTime = example.alarmTime
  data.alarmState = example.alarmState
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data

@router.post("/updateAlarmTime")
def user_login(data : LoginModel):
  print("receive message : " + data.userName +" from : " +data.password + " : " + data.alarmTime)
  example = conn.execute(users.update().where(users.c.name == data.userName).values(alarmTime=data.alarmTime))
  conn.commit()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data

@router.post("/updateAlarmState")
def user_login(data : LoginModel):
  print("receive message : " + data.userName +" from : " +data.password + " : " + data.alarmState)
  example = conn.execute(users.update().where(users.c.name == data.userName).values(alarmState=data.alarmState))
  conn.commit()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data

@router.post("/updateUserInfo")
def user_login(data : LoginModel):
  print("receive message : " + data.userName +" from : " +data.password + " : " + data.alarmTime)
  example = conn.execute(users.update().where(users.c.name == data.userName).values(allergy=data.allergy, disease=data.disease))
  conn.commit()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data

# @router.put("/user/alarm_time")
# def update_alarm_time(alarm_update: LoginModel):
#     session = SessionLocal()
#     try:
#         # 해당 유저를 찾기
#         user = session.query(users).filter(users.c.name == alarm_update.userName).first()
#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")

#         # 알람 시간 업데이트
#         stmt = (
#             update(users).
#             where(users.c.name == alarm_update.userName).
#             values(alarmTime=alarm_update.alarmTime)
#         )
#         session.execute(stmt)
#         session.commit()
#     except:
#         session.rollback()
#         raise
#     finally:
#         session.close()

@router.post("/removeMeal")
def user_getAlarmRecent(data : AlarmModel):
    print("receive message : " + data.userName)
    example = conn.execute(alarms.delete().where(alarms.c.id == data.userMealId))
    print("example : " + str(example))
    conn.commit()
    return data

@router.post("/getAlarmRecent")
def user_getAlarmRecent(data : AlarmModel):
  print("receive message : " + data.userName)
  example = conn.execute(alarms.select().where(alarms.c.userName == data.userName, alarms.c.chkEat == 'BEFORE').order_by(alarms.c.crtTime)).first()
  print("example : " + str(example))
  conn.commit()

  if example is None:
    data.userRecoMeal='null'
    return data
  else:
    data.userMealId = example.id
    data.userRecoMeal = example.recoMeal
    data.userResOfAi = example.resOfAi
    data.userGoodFB = example.goodFB
    data.userBadFB = example.badFB
    data.userChkEat = example.chkEat
    data.userMLD = example.MLD
    data.userAlarmImg = example.alarmImg
  return data

@router.post("/getAlarmAll")
def user_getAlarmAll(data : AlarmModel):
  print("alarm  receive message : " + data.userName)
  example = conn.execute(alarms.select().where(alarms.c.userName == data.userName)).fetchall()
  conn.commit()
  print("alarm example : " + str(example))
  if example is []:
    data.userRecoMeal='null'
    return data
  else:
    # data.userMealInfo = str(example)
    print(data.userMealInfo)
    data.userName=data.userName
    # data.userRecoMeal= '1'
    # data.userResOfAi = '2'
    # data.userChkEat = '3'
    # data.userMLD = '4'
    # data.userAlarmImg = '5'
    print(data.userName)
    # print the JSON data
    json_data_list = list()
    count = 0
    for row in example:
      data2 = {}
      data2['count'] = count
      data2['id'] = row.id
      data2['title'] = row.recoMeal
      data2['goodFB'] = row.goodFB
      data2['badFB'] = row.badFB
      data2['chkEat'] = row.chkEat
      data2['MLD'] = row.MLD
      data2['checked'] = False
      # json_data = json.dumps(data2, ensure_ascii=False)
      # print(json_data) #this is what the script returns 
      json_data_list.append(data2)
      # print("??:"+ " "+ str(json_data_list))
      count += 1
      # print("??:"+ str(row) + " "+ row.MLD)
    data.userMealInfo = json_data_list
  return data

@router.post("/updateAlarmChkEatY")
def user_login(data : AlarmModel):
  print("receive message : " + data.userName + " id : " + data.userMealId )
  example = conn.execute(alarms.update().where(alarms.c.id == data.userMealId ).values(chkEat="Y"))
  conn.commit()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data


@router.post("/updateAlarmChkEatN")
def user_login(data : AlarmModel):
  print("receive message : " + data.userName + " id : " + data.userMealId )
  example = conn.execute(alarms.update().where(alarms.c.id == data.userMealId ).values(chkEat="N"))
  conn.commit()
  print("example : " + str(example))
  if example is None:
    data.userName='null'
  return data

@router.post("/send")
def receive_message(data : Model):
  print("receive message : " + data.content +" from : " +data.userName)
  # example = conn.execute(users.select()).fetchall()
  
  data.content = lcmain.health_agents(data.content)
  try:
    # KJT
    json_dict = json.loads(data.content)
    print("json 로드 가능 상품 추천")
    print(json_dict)
    data.content = json_dict.get("content") # 응답 메시지
    # 데이터 순회 및 insert
    
    for food in json_dict.get("foodList"):
      # b를 수행합니다.
      print(food)
      # food 순회 하며 insert
      # mealForDay '아침', '점심', 저녁
      mealForDay = food.get("mealForDay")
      if (mealForDay == "아침"):
        mealForDay = "M"
      elif(mealForDay == "점심"):
        mealForDay = "L"
      elif(mealForDay == "저녁"):
        mealForDay = "D"
      else:
        mealForDay ="M"
      # productName '상품 이름'
      # 이미지 추가
      example = conn.execute(alarms.insert().values(
        userName=data.userName
        ,alarmState="N"
        ,recoMeal=food.get("productName")
        ,MLD=mealForDay
        ,recoComment=food.get("comment")))
      
    conn.commit()
      
    
  except json.JSONDecodeError:
    print("json 로드 불가능 메시지 바로 반환")
    print(data.content)
    # a를 수행합니다.
  
  
  data.content = data.content.replace('\n', '<br>') # 메시지 html형으로 변환
  data.userName='영양코칭AI'
  
  # print("send result : " + str(data))
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


