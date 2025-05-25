import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def match_census_to_klips(census_csv, klips_vectors_npz, output_matched_csv, output_c2gam_csv):
    print("CENSUS 샘플 ↔ KLIPS 벡터의 common covariates를 cosine similarity로 매칭")

    # CENSUS 데이터 로딩
    census_df = pd.read_csv(census_csv)
    census_df = census_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    # OOM 이슈로 전체의 20%만 랜덤으로 뽑아 샘플링
    
    # KLIPS 임베딩 벡터 로딩
    klips_npz = np.load(klips_vectors_npz)
    klips_vector = klips_npz['sequences']  # [N, T, D]
    klips_targets = klips_npz['targets']
    klips_pids = klips_npz['pids']
    # OOM 이슈로 KLIPS 20% 샘플링
    n_total = klips_vector.shape[0]
    n_sample = max(1, int(n_total * 0.2))
    idx = np.random.default_rng(seed=42).choice(n_total, size=n_sample, replace=False)

    klips_vector = klips_vector[idx]
    klips_targets = klips_targets[idx]
    klips_pids = klips_pids[idx]
    

    # mean pooling
    klips_vectors_flat = klips_vector.mean(axis=1)  # [N, D]

    # 🔧 사용할 CENSUS 공통 covariate 정의
    covariate_cols = ['p_region', 'p_age', 'p_sex', 'p_married', 'p_edu', 'Ind']
    census_vectors = census_df[covariate_cols].fillna(0).values.astype(np.float32)

    # cosine similarity 계산
    similarity_matrix = cosine_similarity(census_vectors, klips_vectors_flat)
    best_match_idx = np.argmax(similarity_matrix, axis=1)

    # matched_sequence_vectors.csv 저장
    matched_df = pd.DataFrame({
        'census_idx': census_df.index,
        'matched_klips_pid': klips_pids[best_match_idx],
        'similarity': similarity_matrix[np.arange(len(census_df)), best_match_idx]
    })
    matched_df.to_csv(output_matched_csv, index=False)

    # 🔧 X1 = 고등교육 여부 (p_edu ≥ 4)
    X1 = (census_df['p_edu'] >= 4).astype(int)

    # 🔧 X2 = covariates 그대로 넣되 이름만 X2_로 재명명
    X2 = census_df[covariate_cols].copy()
    X2.columns = [f'X2_{i+1}' for i in range(len(X2.columns))]

    # Y 및 보조 변수들 생성
    Y = klips_targets[best_match_idx]
    MU0 = Y - 100
    MU1 = Y + 100
    GT = MU1 - MU0
    YC = (Y > np.median(Y)).astype(int)
    S1 = np.zeros_like(Y)

    c2gam_df = pd.concat([
        pd.Series(X1, name="X1"),
        pd.Series(Y, name="Y"),
        pd.Series(YC, name="YC"),
        X2.reset_index(drop=True),
        pd.Series(MU0, name="MU0"),
        pd.Series(MU1, name="MU1"),
        pd.Series(S1, name="S1"),
        pd.Series(GT, name="GT")
    ], axis=1)

    c2gam_df.to_csv(output_c2gam_csv, index=False)
    print(f"Common Covariates 매칭 완료. : {output_c2gam_csv}")

    return matched_df.shape[0], c2gam_df.shape[1]

if __name__ == "__main__":
    match_census_to_klips(
        census_csv="data/raw/census_df.csv",
        klips_vectors_npz="data/klips_census_sequences.npz",
        output_matched_csv="data/matched/census_to_klips_match_info.csv",
        output_c2gam_csv="data/matched/transformed_input_for_c2gam.csv"
    )
