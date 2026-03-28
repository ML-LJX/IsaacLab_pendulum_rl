from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CartDoublePendulumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000 
    save_interval = 50
    experiment_name = "cart_double_pendulum"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128], # 适合低维环境的轻量级网络
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        learning_rate=3e-4, 
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )