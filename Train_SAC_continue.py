# Train_SAC_continue.py

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from end_to_end_nav_env_cnn import EndToEndNavEnv
from custom_cnn import CustomCNNExtractor

policy_kwargs = dict(
    features_extractor_class=CustomCNNExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 128], qf=[256, 128]),
    use_sde=False
)

env = EndToEndNavEnv()

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path='./sac_checkpoints_3/',
    name_prefix='sac_nav'
)

# 저장된 모델 이어서 불러오기
model = SAC.load(
    "./sac_checkpoints_3/sac_nav_10000_steps.zip",
    env=env,
    tensorboard_log="./sac_tensorboard",
    policy_kwargs=policy_kwargs,
    verbose=1
)

# 이어서 학습
model.learn(
    total_timesteps=1000000,  # 추가 학습할 step 수
    callback=checkpoint_callback,
    reset_num_timesteps=False  # ← 이거 반드시 False로 설정해야 이어짐
)

# 최종 저장
model.save("sac_nav_final_continue")
env.close()