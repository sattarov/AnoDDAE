import argparse
import yaml
import numpy as np
import torch

# from src.model import AnomalyDetector
from src.data import load_data, split_data
from src.utils import set_seed, normalize_data, evaluate_anomaly_detection, get_batch_size
from src.model import DDAE, DiffusionScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")
    parser.add_argument('--config', type=str, default='src/config.yaml',
                        help='Path to the config file.')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = config.get('seed', 111)

    # Set seed for reproducibility
    set_seed(seed)

    # Load data
    X, y = load_data(config['data']['path'], config['data']['name'])
    # normalize data
    X = normalize_data(X)
    # split data
    x_train, x_test, y_train, y_test = split_data(X, y, train_setting=config['train']['setting'], random_state=seed)

    # convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize and train model
    model = DDAE(
        input_dim=x_train.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        activation=config['model']['activation'],
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        scheduler=config['diffusion']['scheduler'],
        time_emb_dim=config['diffusion']['time_emb_dim'],
        time_emb_type=config['diffusion']['time_emb_type'],
        epochs=config['train']['epochs'],
        batch_size=get_batch_size(X.shape[0]),
        learning_rate=config['train']['lr'],
        eval_epochs= config['train']['eval_epochs'],
        )
    print("Batch size:", get_batch_size(X.shape[0]))
    model.fit(x_train, x_test, y_train, y_test)

    # Predict anomaly scores
    scores = model.predict(x_test)

    # Evaluate
    results = evaluate_anomaly_detection(scores=scores.numpy(), labels=y_test.numpy())
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()