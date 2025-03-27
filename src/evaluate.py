import time, numpy as np, torch, onnx
from tqdm import tqdm
from onnx2pytorch import ConvertModel
from src.env_utils import Agent, make_env

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_model = onnx.load("models/bc_model.onnx")
    model = ConvertModel(onnx_model).to(device)
    model.eval()
    agent = Agent(model, device)
    scores = []

    for i in tqdm(range(50), desc="Evaluating Episodes", unit="ep"):
        seed = np.random.randint(0, 10000)
        env = make_env(seed=seed, capture_video=True, video_dir="media_eval")
        state, _ = env.reset()
        score = 0.0
        done = False
        step_count = 0
        ep_start_time = time.time()

        step_bar = tqdm(desc=f"Episode {i+1} Steps", unit="step", leave=False)
        while not done:
            step_bar.set_postfix({"Score": f"{score:.2f}", "Step": step_count})
            step_bar.update(1)

            state_np = np.array(state)
            frame = state_np[-1][np.newaxis, ...]
            action = agent.select_action(frame)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            step_count += 1
            done = terminated or truncated
        step_bar.close()
        env.close()
        ep_time = time.time() - ep_start_time
        scores.append(score)
        tqdm.write(f"Episode {i+1}: Score {score:.2f} | Steps {step_count} | Time {ep_time:.2f}s")

    avg_score = np.mean(scores)
    print(f"\nAverage Score over 50 episodes: {avg_score:.2f}")

if __name__ == "__main__":
    evaluate_model()