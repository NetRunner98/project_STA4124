import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class TransformerEncoder(nn.Module):
    """
    KLIPS 시퀀스 데이터를 받아 mean pooling을 통해 고정 길이 벡터로 압축
    """
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # mean pooling
        x = self.output_projection(x)
        return x

if __name__ == "__main__":

    print("Transformer로 KLIPS 시퀀스를 임베딩 중...")

    # 시퀀스 데이터 로드
    npz = np.load("data/klips_census_sequences.npz")
    sequences = torch.tensor(npz["sequences"], dtype=torch.float32)  # [N, T, D]
    pids = npz["pids"]

    # Transformer로 임베딩
    model = TransformerEncoder(input_dim=sequences.shape[2])
    model.eval()  # evaluation 모드
    with torch.no_grad():
        latent_vectors = model(sequences).numpy()  # [N, D]

    # DataFrame 구성: pid + Z1~ZD
    df = pd.DataFrame(latent_vectors, columns=[f"Z{i+1}" for i in range(latent_vectors.shape[1])])
    df.insert(0, "pid", pids)

    # 저장
    output_path = "data/matched/transformer_vectors.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Transformer 임베딩 완료. 벡터 저장 경로: {output_path}")
