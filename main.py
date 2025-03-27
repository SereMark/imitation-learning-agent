import sys
from src.bc import train_bc
from src.dagger import train_with_dagger
from src.evaluate import evaluate_model

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if mode == "bc": train_bc()
    elif mode == "dagger": train_with_dagger()
    elif mode == "eval": evaluate_model()
    else: print("Unknown mode! Usage: python main.py [bc | dagger | eval]"); sys.exit(1)