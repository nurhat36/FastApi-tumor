# test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Kendi main dosyanızın yolunu yazın (örneğin main.py ise böyle kalır)

# FastAPI'nin sanal test sunucusunu başlatıyoruz
client = TestClient(app)


# Test 1: Info API'si çalışıyor mu ve doğru formatta veri dönüyor mu?
def test_get_nifti_info_format():
    # Not: Gerçek bir veritabanı ID'si ve Token yazmalısınız
    headers = {"Authorization": "Bearer BURAYA_GECERLI_BIR_TOKEN_YAZIN"}
    file_id = 1  # Veritabanınızda var olan bir NIfTI dosya ID'si

    response = client.get(f"/api/files/nifti/{file_id}/info", headers=headers)

    # Eğer dosya varsa 200, yoksa 404 dönmeli (500 Internal Error dönerse kod çökmüş demektir!)
    assert response.status_code in [200, 404]

    # Eğer 200 döndüyse, eksen boyutları JSON içinde var mı kontrol et
    if response.status_code == 200:
        data = response.json()
        assert "axial_slices" in data
        assert "coronal_slices" in data
        assert "sagittal_slices" in data


# Test 2: Slice (Kesit) API'si resim döndürüyor mu?
def test_get_nifti_slice():
    headers = {"Authorization": "Bearer BURAYA_GECERLI_BIR_TOKEN_YAZIN"}
    file_id = 1
    slice_index = 50
    axis = "coronal"

    response = client.get(f"/api/files/nifti/{file_id}/slice/{slice_index}?axis={axis}", headers=headers)

    assert response.status_code in [200, 404]
    if response.status_code == 200:
        # Gelen verinin bir PNG resmi olduğunu doğrula
        assert response.headers["content-type"] == "image/png"