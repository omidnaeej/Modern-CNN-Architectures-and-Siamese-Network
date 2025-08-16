import yaml
from logging import raiseExceptions
import random
import torch
from scripts.train import *
from scripts.evaluate import *
from data.data_loader import load_cifar10, load_kaggle_dataset
from utils.visualization import *

def load_config(config_file="config/config.yaml"):
    """
    Load configuration from a YAML file.
    
    :param config_file: Path to the YAML configuration file.
    :return: Dictionary containing the configuration.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def cifar10_classifier():
    print(f"Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = load_cifar10()
    
    config = load_config(config_file="config/config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = train_model(train_loader, val_loader, device=device,
                        block_config=config["block_config"]*8, batch_size =config["batch_size"],
                        num_workers = config["num_workers"], loss = config["loss"], lr=config["lr"], momentum=config["momentum"],
                        gamma=config["gamma"], num_epochs = config["num_epochs"], use_separable_conv=config["use_separable_conv"])
    
    evaluate_model(model, test_loader, device = device, class_names=range(10))

    plot_feature_map_tSNE(model, test_loader, device)

    sample_image, _ = next(iter(train_loader))
    visualize_feature_maps(model, sample_image[10], device)


def face_recognition_siamesenn():
    print(f"Loading Face Recognition dataset...")
    train_loader, val_loader, test_loader = load_kaggle_dataset()
    
    config = load_config(config_file="config/config_snn.yaml")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_losses, val_losses, model = train_siamese_model(train_loader, val_loader, device = device,
                                                          batch_size = config["batch_size"], loss_margin = config["loss_margin"],
                                                          lr = config["lr"], step_size = config["step_size"], gamma = config["gamma"],
                                                          num_epochs = config["num_epochs"])

    analyze_feature_distances(model, test_loader, device)
    
    plot_k_nearest_neighbors(model, test_loader, device)

def random_search(param_distributions, num_iterations, train_loader, val_loader, device):
    """Performs random search to find the best hyperparameters."""
    best_loss = float('inf')
    best_params = {}
    best_model = None

    for _ in range(num_iterations):
        params = {k: random.choice(v) for k, v in param_distributions.items()}
        print(f"Trying parameters: {params}")

        train_losses, val_losses, model = train_siamese_model(train_loader, val_loader, device=device, **params)

        current_loss = val_losses[-1] if val_losses else raiseExceptions("validation losses not found!")

        print(f"Validation Loss: {current_loss:.4f}")
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params
            best_model = model
            print(f"New best loss: {best_loss:.4f} with params: {best_params}")

    return best_params, best_loss, best_model

def face_recognition_siamesenn_random_search():
    print(f"Loading Face Recognition dataset...")
    train_loader, val_loader, test_loader = load_kaggle_dataset()
    
    config = load_config(config_file="config/config_snn.yaml")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    param_distributions = {
        'lr': [0.001, 0.01, 0.1],
        'loss_margin': [0.5, 1.0, 1.5, 2],
        'step_size': [5, 10],
        'gamma': [0.1, 0.05],
        'num_epochs': [15, 20]
    }

    best_params, best_loss, best_model = random_search(param_distributions, num_iterations=5,
                                                    train_loader=train_loader, val_loader=val_loader,
                                                    device=device)

    print(f"\nBest hyperparameters found: {best_params}")
    print(f"Best validation loss: {best_loss:.4f}")


    analyze_feature_distances(best_model, test_loader, device)
    
    plot_k_nearest_neighbors(best_model, test_loader, device)



cifar10_classifier()

face_recognition_siamesenn()

# face_recognition_siamesenn_random_search()
