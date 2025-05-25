import os

class Config:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    data = os.path.join(base_dir, 'D')  # 절대 경로 사용
    experiment_num = 10 # epoch 수
    confounds_num = 5  ########## X2_# 개수 -> bin_feats에 X2_# 인덱스 결정
    bin_feats = [0, 3, 4, 5] 
    config = {
        'encoder_dim_in': confounds_num + 2,
        'encoder_dim_out': 10,
        'encoder_dim_latent': 2,
        'encoder_layer_num': 2, 
        'encoder_lr': 0.003,
        'encoder_wd': 0.0003,
        'vae_epochs': 100,
        'gan_epochs': 200,
        'est_epochs': 100,
        'gen_batch_num': 60,
        'discriminator_dim_in': confounds_num + 2,
        'discriminator_dim_out': 10,
        'discriminator_dim_latent': 1,
        'discriminator_layer_num': 2,
        'discriminator_lr': 0.003,
        'discriminator_wd': 0.0003,
        'generator_dim_in': confounds_num + 2,
        'generator_dim_out': 10,
        'generator_dim_latent': 1,
        'generator_layer_num': 2,
        'generator_lr': 0.005,
        'generator_wd': 0.0005,
        'estimator_dim_in': confounds_num,
        'estimator_dim_out': 10,
        'estimator_dim_latent': 1,
        'estimator_layer_num': 2,
        'est_lr': 0.01,
        'est_wd': 0.001,
        'est_batch_num': 300,
        'IPM_weight': 0.01,
    }
