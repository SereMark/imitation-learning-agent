from glob import glob
import os, torch, numpy as np

class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.files = sorted(glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        if self.files:
            self.counter = max(int(os.path.splitext(os.path.basename(f))[0]) for f in self.files)
        else:
            self.counter = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        with np.load(filename) as data:
            state = data["state"][np.newaxis, ...].astype(np.float32)
            state = state / 255.0
            if self.augment:
                state = self._augment(state)
            action = data["action"]
        return state, int(action.item())

    def _augment(self, state):
        factor = np.random.uniform(0.8, 1.2)
        state = np.clip(state * factor, 0, 1)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.02, state.shape).astype(np.float32)
            state = np.clip(state + noise, 0, 1)
        return state

    def append(self, states, actions):
        for state, action in zip(states, actions):
            self.counter += 1
            filename = os.path.join(self.data_dir, f"{self.counter:06}.npz")
            np.savez_compressed(filename, state=state, action=int(action))
            self.files.append(filename)