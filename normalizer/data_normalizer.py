import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 파일 경로 설정
base_path = "./C2GAM_training/data/"  # 또는 절대 경로로 변경 가능
files = ["D_obs.csv", "D_rep.csv", "D_unsel.csv"]

# 파일 불러오기
dfs = {name: pd.read_csv(os.path.join(base_path, name)) for name in files}

# 정규화할 컬럼들
X_cols = [col for col in dfs["D_rep.csv"].columns if col.startswith("X2_")] + ["Y", "MU0", "MU1"]

# StandardScaler 객체를 D_rep 기준으로 학습
scaler = StandardScaler()
scaler.fit(dfs["D_rep.csv"][X_cols])

# 각 파일에 정규화 적용 후 저장
for name, df in dfs.items():
    df_scaled = df.copy()
    df_scaled[X_cols] = scaler.transform(df[X_cols])
    output_path = os.path.join(base_path, f"normalized_{name}")
    df_scaled.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
