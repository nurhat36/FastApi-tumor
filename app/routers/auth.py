from google.oauth2 import id_token
from google.auth.transport import requests
from pydantic import BaseModel # FastAPI kullanıyorsan veri modeli için
from fastapi import HTTPException # Hata fırlatmak için
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import User
from app.utils.security import (
    get_password_hash,
    verify_password,
    create_access_token
)
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["auth"])


# =========================
# REQUEST MODELS
# =========================
class RegisterRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# 1. React'ten gelecek verinin şemasını belirliyoruz
class GoogleAuthRequest(BaseModel):
    token: str


# 2. React (Frontend) tarafındaki GOOGLE_CLIENT_ID ile birebir AYNI olmalı
GOOGLE_CLIENT_ID = "636599479269-8f0sbt9dpchjfit9so8la30heiqc8ckl.apps.googleusercontent.com"


# Rota adını sadece "/google" yaptık. (Prefix ile birleşince "/auth/google" olacak)
@router.post("/google")
async def google_login(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        # 1. Access Token ile Google'dan kullanıcı bilgilerini iste
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={data.token}"
            )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Geçersiz Google Access Token!")

        idinfo = response.json()

        # Google'dan gelen veriler
        email = idinfo.get('email')
        name = idinfo.get('name', 'Google Kullanıcısı')

        if not email:
            raise HTTPException(status_code=400, detail="Google hesabından email alınamadı.")

        # 2. Veritabanı işlemleri (Burası senin kodunla aynı)
        user = db.query(User).filter(User.username == email).first()  # Genelde username veya email ile eşleşir

        if not user:
            user = User(
                username=email,
                # Email alanın varsa ekle: email=email,
                password="google_sso_user"  # Güvenli bir placeholder
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # 3. Kendi JWT Token'ını üret
        access_token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id
            }
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username
        }

    except Exception as e:
        print(f"HATA google_login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")
# =========================
# REGISTER
# =========================
@router.post("/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    hashed_password = get_password_hash(req.password)

    user = User(
        username=req.username,
        password=hashed_password
        # created_at modelde default var
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "username": user.username
    }


# =========================
# LOGIN
# =========================
@router.post("/token", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }