from gymnasium.spaces import Box
from torch.distributions import Categorical
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, ResizeObservation, GrayScaleObservation, FrameStack
import torch, numpy as np, gymnasium as gym

class Agent:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            logits = self.model(state_t)
            action = Categorical(logits=logits).sample().item()
        return action

class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        return obs[: self.shape[0], : self.shape[1]]

class RecordState(gym.Wrapper):
    def __init__(self, env: gym.Env, reset_clean: bool = True):
        super().__init__(env)
        self.frame_list = []
        self.reset_clean = reset_clean

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.frame_list.append(obs)
        return obs, rew, term, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(obs)
        return obs, info

    def render(self):
        frames = self.frame_list
        self.frame_list = []
        return frames

def make_env(seed=None, capture_video=True, video_dir="media"):
    env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    env = RecordEpisodeStatistics(env)
    if capture_video:
        env = RecordVideo(env, video_dir)
    env = CropObservation(env, (84, 96))
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = RecordState(env, reset_clean=True)
    env = FrameStack(env, 4)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env