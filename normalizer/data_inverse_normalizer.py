import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# === 1. 기준 데이터 불러오기 (D_rep)
rep_df = pd.read_csv("./C2GAM_training/data/D_rep.csv")

# === 2. 생성된 결과 파일 불러오기
generated_df = pd.read_csv("./C2GAM_training/data/D_gen_init.csv")

# === 3. 정규화 대상 변수 설정 (X2_1 ~ X2_6 + Y), X2_7은 제외
scale_cols = [f"X2_{i}" for i in range(1, 7)] + ["Y"]

# === 4. scaler 학습 (D_rep 기준)
scaler = StandardScaler()
scaler.fit(rep_df[scale_cols])

# === 5. 역정규화
generated_df_scaled = generated_df.copy()
generated_df_scaled[scale_cols] = scaler.inverse_transform(generated_df[scale_cols])

# === 6. 정수형 변수 보정 (X2_1~X2_6, X2_7 모두 정수형이라고 가정)
int_cols = [f"X2_{i}" for i in range(1, 8)]
for col in int_cols:
    if col in generated_df_scaled.columns:
        generated_df_scaled[col] = generated_df_scaled[col].round().astype(int)

# === 7. YC 생성 (rep_df의 Y 중간값 기준)
median_Y = rep_df["Y"].median()
generated_df_scaled["YC"] = (generated_df_scaled["Y"] > median_Y).astype(int)

# === 8. 컬럼 순서 재정렬
ordered_cols = ["X1", "Y", "YC"] + [f"X2_{i}" for i in range(1, 8)]
ordered_cols = [col for col in ordered_cols if col in generated_df_scaled.columns]
generated_df_scaled = generated_df_scaled[ordered_cols]
# X1 값을 0 혹은 1에 반올림
if "X1" in generated_df_scaled.columns:
    generated_df_scaled["X1"] = (generated_df_scaled["X1"] >= 0.5).astype(int)


# === 9. 저장
output_path = "./C2GAM_training/data/D_gen.csv"
generated_df_scaled.to_csv(output_path, index=False)
print(f"복원된 결과 저장 완료: {output_path}")
