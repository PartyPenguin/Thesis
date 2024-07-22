import src.prepare as prepare
import src.train as train
import src.evaluate as evaluate
import wandb
import yaml

# Load config from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)


def pipeline(config):
    with wandb.init(project="GNN_BC", config=config):
        config = wandb.config

        # Prepare data
        prepare.prepare(config)

        # Train the model
        train.train(config)

        # Evaluate the model
        evaluate.evaluate(config)


if __name__ == "__main__":
    pipeline(config)
