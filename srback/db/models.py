from sqlalchemy import Column, Table
from sqlalchemy.sql.sqltypes import INTEGER, String
from srback.db.db import meta, engine
users = Table('user', meta,
              Column('id', INTEGER(), primary_key=True),
              Column('name', String(255)),
              Column('pw', String(255)),
              Column('message', String(255)),
              Column('age', INTEGER()),
              )
meta.create_all(engine)