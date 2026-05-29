import pymysql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

def create_database_if_not_exists():
    """
    Connect to MySQL server using pymysql and create the target database if it doesn't exist.
    """
    try:
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD
        )
        try:
            with connection.cursor() as cursor:
                # Use backticks for database name to avoid syntax errors with dashes or spaces
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.MYSQL_DATABASE}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                )
            connection.commit()
            print(f"MySQL Database '{settings.MYSQL_DATABASE}' initialized successfully or already exists.")
        finally:
            connection.close()
    except Exception as e:
        print(f"Warning: Connection to MySQL server failed. Error details: {e}")
        print("Please ensure your MySQL service is running and credentials in .env are correct.")

# Auto-create database if possible before initializing engine
create_database_if_not_exists()

# Initialize SQLAlchemy
engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,  # Test connections before executing queries
    pool_recycle=3600,   # Prevent MySQL idle connection timeouts
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency generator for obtaining DB sessions in FastAPI routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
