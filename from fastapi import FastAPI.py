from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
MODEL_PATH = "price_model.joblib"


# ---------------------------------------------------------
# 모델 불러오기
# ---------------------------------------------------------
saved = joblib.load(MODEL_PATH)

rf_model = saved["rf_model"]
feature_columns = saved["feature_columns"]
target_col = saved["target_col"]


# ---------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------
app = FastAPI(
    title="중고차 가격 예측 API",
    description="연식, 주행거리, 배기량 등의 정보를 이용해 중고차 가격 비율을 예측합니다.",
    version="1.0.0",
)


# ---------------------------------------------------------
# 입력 데이터 형식
# ---------------------------------------------------------
class CarInput(BaseModel):
    주행거리: float
    연식: float
    배기량: float
    소유자변경횟수: float = 0

    사고상세_전손: float = 0
    사고상세_침수전손: float = 0
    사고상세_침수분손: float = 0
    사고상세_도난: float = 0

    보험_내차피해_횟수: float = 0
    보험_내차피해_가격: float = 0

    사고상세_타차가해_횟수: float = 0
    보험_타차피해_가격: float = 0


# ---------------------------------------------------------
# 기본 확인용 API
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "중고차 가격 예측 API가 정상적으로 실행 중입니다.",
        "target": target_col,
    }


# ---------------------------------------------------------
# 예측 API
# ---------------------------------------------------------
@app.post("/predict")
def predict(car: CarInput):

    # 입력값을 DataFrame으로 변환
    data = {
        "주행거리": car.주행거리,
        "연식": car.연식,
        "배기량": car.배기량,
        "소유자변경횟수": car.소유자변경횟수,

        "사고상세_전손": car.사고상세_전손,
        "사고상세_침수전손": car.사고상세_침수전손,
        "사고상세_침수분손": car.사고상세_침수분손,
        "사고상세_도난": car.사고상세_도난,

        "보험_내차피해(횟수)": car.보험_내차피해_횟수,
        "보험_내차피해(가격)": car.보험_내차피해_가격,

        "사고상세_타차가해(횟수)": car.사고상세_타차가해_횟수,
        "보험_타차피해(가격)": car.보험_타차피해_가격,
    }

    input_df = pd.DataFrame([data])

    # 학습 당시의 feature_columns와 동일하게 맞춤
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 랜덤포레스트 예측
    prediction = rf_model.predict(input_df)[0]

    return {
        "prediction": float(prediction),
        "target": target_col,
        "message": "예측이 완료되었습니다."
    }


# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )