from sqlalchemy import Column, Table
from sqlalchemy.sql.sqltypes import INTEGER, String
from srback.db.db import meta, engine
users = Table('user', meta,
              Column('id', INTEGER(), primary_key=True),
              Column('name', String(255)),
              Column('pw', String(255)),
              Column('message', String(255)),
              Column('age', INTEGER()),
              Column('alarmTime', String(32)),
              Column('alarmState', String(32)),
              )

alarms = Table('alarm', meta,
              Column('id', INTEGER(), primary_key=True),
              Column('userName', String(255)),
              Column('alarmState', String(255)),
              Column('recoMeal', String(255)),
              Column('resOfAi', String(255)),
              Column('chkEat', String(255)),
              Column('MLD', String(255)),
              Column('alarmImg', String(255)),
              Column('alarmTime', String(255)),
              Column('crtTime', String(255)),
              Column('chgTime', String(255)),
              )


meta.create_all(engine)