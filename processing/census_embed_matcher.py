import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def match_census_to_klips(census_csv, klips_covariate_csv, output_matched_csv, output_c2gam_csv):
    print("CENSUS 샘플 ↔ KLIPS 벡터의 common covariates를 cosine similarity로 매칭")

    # KLIPS 복원된 covariates 로딩 (7차원)
    klips_df = pd.read_csv(klips_covariate_csv)
    klips_vectors = klips_df[['year', 'p_region', 'p_age', 'p_sex', 'p_married', 'p_edu', 'Ind']].values
    klips_pids = klips_df["pid"].values

    # KLIPS의 20%만 샘플링
    rng = np.random.default_rng(seed=42)
    total_klips = len(klips_df)
    sample_size = max(1, int(total_klips * 0.1))
    sample_indices = rng.choice(total_klips, size=sample_size, replace=False)

    klips_vectors = klips_vectors[sample_indices]
    klips_pids = klips_pids[sample_indices]

    # CENSUS 데이터 로딩
    census_df = pd.read_csv(census_csv)
    census_df = census_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    covariate_cols = ['year', 'p_region', 'p_age', 'p_sex', 'p_married', 'p_edu', 'Ind']

    # 정수형 변수 보정
    for col in ['year', 'p_region', 'Ind']:
        census_df[col] = census_df[col].round().astype(int)

    census_vectors = census_df[covariate_cols].fillna(0).values.astype(np.float32)

    # similarity and distance matrix
    cos_sim = cosine_similarity(census_vectors, klips_vectors)  # [N x M]
    euc_dist = pairwise_distances(census_vectors, klips_vectors, metric='euclidean')  # [N x M]

    # Euclidean distance 정규화 (0~1 사이)
    scaler = MinMaxScaler()
    euc_dist_norm = scaler.fit_transform(euc_dist)

    # 하이브리드 스코어 계산
    alpha = 0.5  # cosine 가중치
    hybrid_score = alpha * cos_sim - (1 - alpha) * euc_dist_norm

    # best match index (가장 높은 hybrid score)
    best_match_idx = np.argmax(hybrid_score, axis=1)
    matched_pids = klips_pids[best_match_idx].astype(int)

    print("matched_pids 고유값 수:", len(set(matched_pids)))
    print("고유 pid 예시:", set(matched_pids[:10]))

    # 매칭 결과 저장 (hybrid score 기준)
    matched_df = pd.DataFrame({
        'census_idx': census_df.index,
        'matched_klips_pid': matched_pids,
        'hybrid_score': hybrid_score[np.arange(len(census_df)), best_match_idx],
        'cosine_similarity': cos_sim[np.arange(len(census_df)), best_match_idx],
        'euclidean_distance': euc_dist[np.arange(len(census_df)), best_match_idx]
    })
    matched_df.to_csv(output_matched_csv, index=False)

    matched_df.to_csv(output_matched_csv, index=False)

    # X1 = 고등교육 여부
    X1 = (census_df['p_edu'] >= 4).astype(int)

    # X2 = covariates 복사
    X2 = census_df[covariate_cols].copy()
    X2.columns = [f'X2_{i+1}' for i in range(len(X2.columns))]

    # Y = KLIPS에서 가져온 평균 wage
    wage_df = pd.read_csv("data/matched/klips_pid_mean_wage.csv")
    wage_map = dict(zip(wage_df["pid"], wage_df["mean_wage"]))
    Y = np.array([wage_map.get(pid, 0) for pid in matched_pids])

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
        klips_covariate_csv="data/matched/klips_covariate_recon.csv",
        output_matched_csv="data/matched/census_to_klips_match_info.csv",
        output_c2gam_csv="data/matched/transformed_input_for_c2gam.csv"
    )
