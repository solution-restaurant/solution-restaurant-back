
import os
from sqlalchemy import create_engine, MetaData
print(os.getenv("DB_PATH"))
engine = create_engine(os.getenv("DB_PATH"))
meta = MetaData()
conn = engine.connect()