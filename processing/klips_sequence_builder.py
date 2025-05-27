import pandas as pd
import numpy as np
import os

def build_klips_sequences(input_csv, output_path, sequence_length=5):
    """
    KLIPS 원본 csv 데이터를 시계열 데이터로 바꿔 Transformer 입력용 텐서로 바꾸는 코드
    """

    print("KLIPS panel data를 transformer에 넣을 수 있도록 처리\n")

    df = pd.read_csv(input_csv)

    # 기존 pid column 사용
    if 'pid' not in df.columns:
        raise ValueError("입력 CSV에 'pid' 컬럼이 존재해야 합니다.")

    # 사용할 Covariates (CENSUS와 매칭되는 변수 포함)
    covariate_cols = ['year', 'p_region', 'p_age', 'p_sex', 'p_married', 'p_edu', 'Ind']
    time_col = 'year'
    id_col = 'pid'
    target_col = 'p_wage'

    grouped = df.groupby(id_col)
    sequences, targets, valid_ids, x1_values = [], [], [], []

    for pid, group in grouped:
        group_sorted = group.sort_values(time_col)
        X = group_sorted[covariate_cols].values

        if X.shape[0] == 0:
            continue

        if X.shape[0] >= sequence_length:
            seq = X[:sequence_length]
        else:
            padding = np.zeros((sequence_length - X.shape[0], len(covariate_cols)))
            seq = np.vstack([X, padding])

        imcome_seq = group_sorted[target_col].values
        income_mean = np.mean(imcome_seq)
        x1 = 1 if group_sorted['p_edu'].max() >= 4 else 0

        sequences.append(seq)
        targets.append(income_mean)
        valid_ids.append(pid)
        x1_values.append(x1)

    # 저장
    np.savez_compressed(output_path,
                        sequences=np.array(sequences, dtype=np.float32),
                        targets=np.array(targets, dtype=np.float32),
                        pids=np.array(valid_ids),
                        x1=np.array(x1_values, dtype=np.int32))

    return len(sequences), len(covariate_cols)

if __name__ == "__main__":
    input_csv = "data/raw/klips_df.csv"
    output_npz = "data/klips_census_sequences.npz"

    n_seq, n_feat = build_klips_sequences(
        input_csv=input_csv,
        output_path=output_npz,
        sequence_length=5
    )

    print(f"Panel data num: {n_seq} | Covariates num: {n_feat}\n")
