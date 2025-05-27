import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim=64, output_dim=7):
        super(TransformerDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__":
    print("학습된 Transformer 모델을 로드하여 KLIPS 시퀀스 임베딩 및 복원 중...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    npz = np.load("data/klips_census_sequences.npz")
    sequences = torch.tensor(npz["sequences"], dtype=torch.float32).to(device)  # [N, T, D=7]
    pids = npz["pids"]

    # 모델 정의 및 가중치 로드
    encoder = TransformerEncoder(input_dim=sequences.shape[2]).to(device)
    decoder = TransformerDecoder(model_dim=64, output_dim=sequences.shape[2]).to(device)

    encoder.load_state_dict(torch.load("data/matched/transformer_encoder.pt", map_location=device))
    decoder.load_state_dict(torch.load("data/matched/transformer_decoder.pt", map_location=device))

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        latent_vectors = encoder(sequences)  # [N, D]
        reconstructed_covariates = decoder(latent_vectors).cpu()  # [N, 7]

    # 복원된 covariates 저장
    reconstructed_df = pd.DataFrame(reconstructed_covariates.numpy(),
                                    columns=['year', 'p_region', 'p_age', 'p_sex', 'p_married', 'p_edu', 'Ind'])
    reconstructed_df.insert(0, "pid", pids)
    reconstructed_df["year"] = reconstructed_df["year"] /100 + 2000
    os.makedirs("data/matched", exist_ok=True)
    reconstructed_df.to_csv("data/matched/klips_covariate_recon.csv", index=False)

    # latent 벡터 저장
    latent_df = pd.DataFrame(latent_vectors.cpu().numpy(), columns=[f"Z{i+1}" for i in range(latent_vectors.shape[1])])
    latent_df.insert(0, "pid", pids)
    latent_df.to_csv("data/matched/transformer_vectors.csv", index=False)

    print("복원 및 임베딩 벡터 저장 완료 : data/matched/klips_covariate_recon.csv")
