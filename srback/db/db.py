
import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker

print(os.getenv("DB_PATH"))
engine = create_engine(os.getenv("DB_PATH"))
meta = MetaData()
conn = engine.connect()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)