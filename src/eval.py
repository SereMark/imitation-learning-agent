import onnx
import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from onnx2pytorch import ConvertModel
from torch.distributions.categorical import Categorical

class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation[:self.shape[0], :self.shape[1]]

class Agent():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def select_action(self, state):        
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            logits = self.model(state)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = Categorical(logits=logits)
            return probs.sample().cpu().numpy()[0]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvertModel(onnx.load("models/dagger_model.onnx"))
    model.eval()
    model = model.to(device)
    agent = Agent(model, device)
    scores = []
    for _ in range(50):
        seed_episode = np.random.randint(1e7)
        env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CropObservation(env, (84, 96))
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        state, _ = env.reset(seed=seed_episode)
        env.action_space.seed(seed_episode)
        env.observation_space.seed(seed_episode)
        score, done = 0, False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        env.close()
        scores.append(score)
    print(np.mean(scores))