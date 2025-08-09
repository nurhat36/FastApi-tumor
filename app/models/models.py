from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)  # hash saklanacak
    created_at = Column(DateTime, default=datetime.utcnow)

    masks = relationship("Mask", back_populates="owner")


class Mask(Base):
    __tablename__ = "masks"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    mask_data = Column(LargeBinary, nullable=False)  # PNG binary
    created_at = Column(DateTime, default=datetime.utcnow)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="masks")
