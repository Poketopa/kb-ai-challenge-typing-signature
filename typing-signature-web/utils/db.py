import os
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime
from sqlalchemy.dialects.mysql import BLOB, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import uuid

# 환경변수에서 MySQL 접속 정보 읽기
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'password')
MYSQL_HOST = os.getenv('MYSQL_HOST', 'db')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DB = os.getenv('MYSQL_DB', 'users_db')

DB_PATH = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4'
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(VARCHAR(50), unique=True, nullable=False)
    embedding = Column(BLOB, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_verified_at = Column(DateTime, nullable=True)

engine = create_engine(DB_PATH, echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    # 단순화를 위해 매 실행 시 스키마를 재생성합니다. (데이터 보존 고려하지 않음)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
