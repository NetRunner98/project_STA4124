import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import expit
import os

def generate_selection_variable(input_csv, output_dir, rep_ratio=0.1, unsel_ratio=0.2, random_state=42):
    print("collider bias를 S1로 생성하여 주입하고, KLIPS obs/rep/unsel 데이터 분리")

    np.random.seed(random_state)
    df = pd.read_csv(input_csv)
    df = df.sample(frac=0.005, random_state=random_state).reset_index(drop=True)
    ######### 전체의 몇퍼센트만 샘플링 할지 결정
    
    
    # 필수 컬럼 확인
    required_cols = ['X1', 'Y', 'MU0', 'MU1', 'GT', 'YC']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 '{col}' 이(가) {input_csv}에 없습니다.")

    # 공변량 추출
    X_cols = [col for col in df.columns if col.startswith("X2_")]
    X = df[X_cols].values
    T = df["X1"].values
    Y = df["Y"].values

    # S1 생성 (collider 구조)
    w = np.random.randn(X.shape[1])
    epsilon = np.random.randn(len(df))
    logits = 0.01*(Y - 3 * T + X @ w + epsilon) # 관측 데이터가 없는걸 방지
    S = np.random.binomial(n=1, p=expit(logits))
    df["S1"] = S

    # Dobs: S=1
    df_Dobs = df[df["S1"] == 1].copy()

    # Drep: 무작위 표본 (최소 1개 보장)
    n_rep = max(1, int(len(df) * rep_ratio))
    df_Drep = df.sample(n=n_rep, random_state=random_state).copy()

    # Dunsel: S=0 중에서 무작위 (없으면 에러)
    df_s0 = df[df["S1"] == 0]
    if len(df_s0) == 0:
        raise ValueError("S=0 샘플이 없어 unselected set을 생성할 수 없습니다.")
    n_unsel = max(1, int(len(df_s0) * unsel_ratio))
    df_Dunsel = df_s0.sample(n=n_unsel, random_state=random_state).copy()

    os.makedirs(output_dir, exist_ok=True)

    cols_to_save = ['X1', 'Y', 'YC'] + X_cols + ['MU0', 'MU1', 'S1', 'GT']
    df_Dobs[cols_to_save].to_csv(os.path.join(output_dir, "D_obs.csv"), index=False)
    df_Drep[cols_to_save].to_csv(os.path.join(output_dir, "D_rep.csv"), index=False)
    df_Dunsel[cols_to_save].to_csv(os.path.join(output_dir, "D_unsel.csv"), index=False)

    #print(f"Total sample num: {len(df)}")
    print(f"[S=1] D_obs num: {len(df_Dobs)}")
    print(f"[random {int(rep_ratio * 100)}%] D_rep num: {len(df_Drep)}")
    print(f"[S=0 중 {int(unsel_ratio * 100)}%] unselected sample num: {len(df_Dunsel)}")

if __name__ == "__main__":
    generate_selection_variable(
        input_csv="data/matched/transformed_input_for_c2gam.csv",
        output_dir="C2GAM_master/data"
    )
