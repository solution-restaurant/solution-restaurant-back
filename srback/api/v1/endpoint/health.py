from typing import Any, List, Union
from fastapi import APIRouter, Depends, HTTPException
from srback.llm.lclogic import lcmain
from srback.llm.lclogic.agents.health_agent import getGoodFb, getbadFb

import json
import urllib.request
from srback.core.config import settings
from pydantic import BaseModel

from srback.db.models import users
from srback.db.models import alarms
from srback.db.models import meals
from srback.db.db import conn

from srback.db.db import SessionLocal
from sqlalchemy import update, select

import srback.api.v1.endpoint.userProperties as globalUser

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
  globalUser.globalUserName = data.userName
  example = conn.execute(users.select().where(users.c.name == data.userName)).first()
  if example  is not None:
    globalUser.globalUserAllegy = example.allergy
    globalUser.globalUserDisease = example.disease
    
  conn.commit()
  
  data.content = lcmain.health_agents(data.content)
  try:
    json_dict = json.loads(data.content)
    # KJT 위에서 에러 안나면 상품 추천이라고 간주
    print("json 로드 가능 상품 추천 Start")
    # data.content = json_dict.get("content") # 응답 메시지
    # food 순회 하며 content 생성
    data.content = create_html_contents_from_db(data.userName, json_dict["foodList"], json_dict["reviews"])
    print("json 로드 가능 상품 추천 End")
    
  except Exception as e:
    print(f"Error: {e}")
    print("json 로드 불가능 메시지 바로 반환 Start")
    print(data.content)
    print("json 로드 불가능 메시지 바로 반환 End")
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
























# 함수 분리 KJT Start

def create_html_contents_from_db(userName, foodList, reviews):
  
    # 식사 시간 리스트
    meal_for_day = ['아침', '점심', '저녁']

    # 결과를 저장할 빈 문자열
    result = ''

    # foodList를 순회하면서 HTML 내용 생성
    for i, food in enumerate(foodList):
        # id에 해당하는 행을 meal 테이블에서 조회
        stmt = select(
            meals.c.id,
            meals.c.product_name, 
            meals.c.comment, 
            meals.c.calorie, 
            meals.c.img_url
        ).where(meals.c.id == food['id'])
        
        # 쿼리 실행
        queryResult = conn.execute(stmt)
        row = queryResult.fetchone()

        # 결과 받기 (여기서는 샘플 데이터 사용)

        if row is not None:
            # comment 에 대하여 goodChat, badChat 사용해서 output 으로 goodFb, badFb 선택
            goodFb = getGoodFb(row.comment)
            badFb = getbadFb(row.comment)
            
            print("장점 : " + goodFb + "\n" + "단점 : " + badFb)

            # HTML 생성
            html_content = create_html_content(
                row.id,
                meal_for_day[i % len(meal_for_day)], 
                row.product_name, 
                row.comment, 
                row.calorie, 
                row.img_url, 
                reviews if i == len(foodList) - 1 else ""  # 마지막 아이템에만 reviews 추가
            )
            # 식단 기록에 추가 데이터 순회 및 insert Start
            try:
              example = conn.execute(alarms.insert().values(
                userName=userName
                ,alarmState="N"
                ,recoMeal=row.product_name
                ,MLD=meal_for_day[i % len(meal_for_day)]
                ,alarmImg=row.img_url
                ,recoComment=row.comment
                ,goodFB=goodFb
                ,badFB=badFb
              ))
              # 식단 기록에 추가 데이터 순회 및 insert End
            except Exception as e:
              print(e)
            
            result += html_content
    conn.commit()
    
    # 결과 반환
    return result
      

# create_html_content 함수 (여기서는 샘플 데이터를 사용합니다)
# 1차
# def create_html_content(meal_for_day, product_name, comment, calorie, img_url, reviews):
#     html_content = f'''
#     {meal_for_day} <br>
#     <img src="{img_url}"></img><br>
#     상품 이름 : {product_name}<br>
#     추천 이유 : {comment}<br>
#     칼로리 : {calorie} kcal <br>
#     <br><br>
#     '''
#     if reviews:
#         html_content += f'<br><br>{reviews}'
#     return html_content
# 2차
# def create_html_content(meal_for_day, product_name, comment, calorie, img_url, reviews):
#     html_content = f'''
#     <div style="border:1px solid black; padding:20px; margin:10px">
#         <h2>{meal_for_day}</h2>
#         <img src="{img_url}" width="300px" height="200px">
#         <h3>상품 이름 : {product_name}</h3>
#         <p>추천 이유 : {comment}</p>
#         <p>칼로리 : {calorie} kcal </p>
#     </div>
#     '''
#     if reviews:
#         html_content += f'<div style="padding:20px; margin:10px"><h3>리뷰:</h3><p>{reviews}</p></div>'
#     return html_content
# 3차
# def create_html_content(meal_for_day, product_name, comment, calorie, img_url, reviews):
#     html_content = f'''
#     <div style="border:1px solid black; padding:20px; margin:10px; max-width:100%">
#         <h2>{meal_for_day}</h2>
#         <img src="{img_url}" style="max-width:100%; height:auto">
#         <h3>상품 이름 : {product_name}</h3>
#         <p>추천 이유 : {comment}</p>
#         <p>칼로리 : {calorie} kcal </p>
#     </div>
#     '''
#     if reviews:
#         html_content += f'<div style="padding:20px; margin:10px; max-width:100%"><h3>리뷰:</h3><p>{reviews}</p></div>'
#     return html_content
# 4차
# def create_html_content(meal_for_day, product_name, comment, calorie, img_url, reviews):
#     html_content = f'''
#     <div style="border:1px solid #f2f2f2; padding:20px; margin:10px; max-width:100%; font-family: Arial, sans-serif;">
#         <h2 style="color: #4CAF50;">{meal_for_day}</h2>
#         <div style="display: flex; align-items: center;">
#             <img src="{img_url}" style="max-width:30%; height:auto; margin-right: 20px;">
#             <div>
#                 <h3 style="margin-bottom: 10px;">상품 이름 : {product_name}</h3>
#                 <p style="margin-bottom: 10px;">추천 이유 : {comment}</p>
#                 <p style="color: #FF5722;">칼로리 : {calorie} kcal </p>
#             </div>
#         </div>
#     </div>
#     '''
#     if reviews:
#         html_content += f'<div style="border:1px solid #f2f2f2; padding:20px; margin:10px; max-width:100%; font-family: Arial, sans-serif;"><h3 style="color: #4CAF50;">리뷰:</h3><p>{reviews}</p></div>'
#     return html_content

# 5차 
# def create_html_content(meal_for_day, product_name, comment, calorie, img_url, reviews):
#     html_content = f'''
#     <div style="border:1px solid #f2f2f2; padding:10px; margin:5px; max-width:100%; font-family: Arial, sans-serif;">
#         <h2 style="color: #4CAF50; margin-bottom: 10px;">{meal_for_day}</h2>
#         <div style="display: flex; align-items: flex-start;">
#             <img src="{img_url}" style="max-width:30%; height:auto; margin-right: 20px;">
#             <div>
#                 <h3 style="margin-bottom: 10px;">{product_name}</h3>
#                 <p style="margin-bottom: 10px;">{comment}</p>
#                 <p style="color: #FF5722;">{calorie} kcal</p>
#             </div>
#         </div>
#     </div>
#     '''
#     if reviews:
#         html_content += f'<div style="border:1px solid #f2f2f2; padding:10px; margin:5px; max-width:100%; font-family: Arial, sans-serif;"><h3 style="color: #4CAF50; margin-bottom: 10px;">리뷰:</h3><p>{reviews}</p></div>'
#     return html_content
# 최종 
def create_html_content(id, meal_for_day, product_name, comment, calorie, img_url, reviews):
    html_content = f'''
    <div style="display: inline-block; max-width: 100%; margin: 10px; text-align: center; font-family: Arial, sans-serif;">
        <div style="border: 1px solid #ccc; padding: 5px;">
            <a href="https://www.greating.co.kr/market/marketDetail?itemId={id}" return false;">
                <img src="{img_url}" alt="Product Image" style="max-width: 100%; height: auto;">
            </a>
        </div>
        <h3>{meal_for_day}</h3>
        <h4>{product_name}</h4>
        <p>{comment}</p>
        <p>칼로리: {calorie} kcal</p>
    </div>
    '''
    if reviews:
        html_content += f'''
        <div text-align: center;>
            <p>{reviews}</p>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <a href="#/admin/user" style="display: inline-block; flex: 1; color: black; background-color: #e8e8e8; text-align: center; padding: 10px; text-decoration: none; font-size: 14px; margin: 4px 2px; cursor: pointer; border-radius: 12px; max-width: 50%; box-sizing: border-box;">식습관<br>수정하기</a>
            <a href="#/admin/table-list" style="display: inline-block; flex: 1; color: black; background-color: #e8e8e8; text-align: center; padding: 10px; text-decoration: none; font-size: 14px; margin: 4px 2px; cursor: pointer; border-radius: 12px; max-width: 50%; box-sizing: border-box;">추천 식단<br>보러가기</a>
        </div>
        '''
    return html_content
# 함수 호출


# 함수 분리 KJT End