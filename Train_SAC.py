# Train_SAC.py

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from end_to_end_nav_env_cnn import EndToEndNavEnv
from custom_cnn import CustomCNNExtractor

policy_kwargs = dict(
    features_extractor_class=CustomCNNExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 128], qf=[256, 128])  # SAC에 맞게 조정
)

env = EndToEndNavEnv()

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path='./sac_checkpoints_3/',
    name_prefix='sac_nav'
)

model = SAC(
    policy='MultiInputPolicy',
    env=env,
    learning_rate=3e-4,
    buffer_size=100000,     
    batch_size=256,
    tau=0.005,                
    gamma=0.99,
    train_freq=1,            
    gradient_steps=1,         
    ent_coef='auto',      
    verbose=1,
    tensorboard_log="./sac_tensorboard",
    policy_kwargs=policy_kwargs
)

model.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback
)

model.save("sac_nav_final")
env.close()
