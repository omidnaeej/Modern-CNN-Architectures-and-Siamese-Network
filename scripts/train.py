import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from models.model import *
from utils.visualization import *

def train_model(train_loader, val_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                block_config=['A']*8, batch_size = 64,
                num_workers = 2, loss = "CE", lr=0.1, momentum=0.5,
                gamma=0.99, num_epochs = 30, use_separable_conv=False, save_path="models/saved_models/model_weights.pth"):

    model = BaseCNN(block_config, use_separable_conv=use_separable_conv)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if loss == "CE":
        criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Calculate epoch loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        scheduler.step()

    end_time = time.time()
    training_time = end_time - start_time

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Training time: {training_time:.2f} seconds")

    plot_loss_and_acc(train_losses, val_losses, train_accuracies, val_accuracies)
    torch.save(model.state_dict(), save_path)
    return model

def train_siamese_model(train_loader, val_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        batch_size = 64, loss_margin=1, lr=0.001,
                        step_size=5, gamma=0.1, num_epochs = 20):

    model = SiameseNetwork().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion = ContrastiveLoss(margin=loss_margin)

    train_losses = []
    val_losses = []

    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Calculate epoch loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        scheduler.step()

    end_time = time.time()
    training_time = end_time - start_time

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Training time: {training_time:.2f} seconds")

    plot_loss_curve(train_losses, val_losses, loss="Contrastive")
    return train_losses, val_losses, model
