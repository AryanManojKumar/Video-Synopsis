from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.sql import func
from db.session import Base
import uuid

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    video_path = Column(String)
    result_url = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
