#!/bin/bash

echo "[1] KLIPS 시퀀스 빌드"
python preprocessing/klips_sequence_builder.py

echo "[2] Transformer AutoEncoder 학습 및 임베딩"
python preprocessing/transformer_autoencoder.py
python preprocessing/transformer_klips.py

echo "[3] CENSUS ↔ KLIPS cosine 매칭"
python preprocessing/census_embed_matcher.py

echo "[4] S 변수 생성 및 D_obs, D_rep, D_unsel 분리"
python selection/make_selection.py

echo "[5] C2GAM 학습 및 평가"
python c2gam_training/main.py

echo "전체 파이프라인 완료!"
