import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

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
    def __init__(self, model_dim=64, output_dim=7):  # 7차원 복원
        super(TransformerDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__":
    print("KLIPS 시퀀스 데이터를 학습 중...")

    # 데이터 로드
    npz = np.load("data/klips_census_sequences.npz")
    sequences = torch.tensor(npz["sequences"], dtype=torch.float32)  # [N, T, D=7]
    target_covariates = sequences.mean(dim=1)  # 복원 타겟: 시계열 평균값 [N, 7]
    pids = npz["pids"]

    # 모델 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TransformerEncoder(input_dim=sequences.shape[2]).to(device)
    decoder = TransformerDecoder(model_dim=64, output_dim=7).to(device)

    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr=1e-3)
    loss_fn = nn.MSELoss()

    # 학습
    n_epochs = 200
    batch_size = 128
    num_batches = int(np.ceil(len(sequences) / batch_size))

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        encoder.train()
        decoder.train()

        for i in range(num_batches):
            x_batch = sequences[i * batch_size:(i + 1) * batch_size].to(device)
            y_batch = target_covariates[i * batch_size:(i + 1) * batch_size].to(device)

            optimizer.zero_grad()
            latent = encoder(x_batch)
            y_pred = decoder(latent)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.4f}")

    # 학습 완료 후 저장
    torch.save(encoder.state_dict(), "data/matched/transformer_encoder.pt")
    torch.save(decoder.state_dict(), "data/matched/transformer_decoder.pt")

    print("모델 학습 및 저장 완료.")
