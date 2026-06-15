import sys
from model import load_model, predict_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)

    model, cfg = load_model("deploy_model.pt", device="cpu")
    verdict, p_fake = predict_file(sys.argv[1], model, cfg)

    print(f"File:        {sys.argv[1]}")
    print(f"Prediction:  {verdict}")
    print(f"P(deepfake): {p_fake:.4f}   (threshold {cfg['threshold']:.3f})")