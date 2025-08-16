import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
import seaborn as sns

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms

from models.model import *

def plot_random_images(dataset, num_images=50):
    fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 20))
    for i in range(num_images):
        random_index = random.randint(0, len(dataset) - 1)
        image, label = dataset[random_index]

        # Convert image to numpy array and clip values to [0, 1]
        image = np.clip(image.numpy().transpose((1, 2, 0)), 0, 1)

        row = i // 5
        col = i % 5
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Label: {label}")
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()    
    
def valdata_target_distribution(val_dataset):
    # Create a dictionary to store the class counts
    class_counts = {}
    for _, label in val_dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Extract the class labels and counts
    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    # Plot the distribution
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Instances")
    plt.title("Distribution of Classes in Validation Dataset")

    # Add the exact number of instances above each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.show()

def plot_augmentation_results(train_dataset):
    # Select a random image from the dataset
    image_index = random.randint(0, len(train_dataset) - 1)
    image, label = train_dataset[image_index]

    image_pil = transforms.ToPILImage()(image)

    # Define color jitter and random affine transformations
    color_jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
    affine_random = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))

    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    fig.suptitle("Color Jitter")

    for i in range(10):
        image_pil = color_jitter(image_pil)
        axes[i].imshow(image_pil)
        axes[i].set_title(f"Augmented {i+1}")
        axes[i].axis('off')
    plt.show()


    image_pil = transforms.ToPILImage()(image)

    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    fig.suptitle("Random Affine")

    for i in range(10):
        image_pil = affine_random(image_pil)
        axes[i].imshow(image_pil)
        axes[i].set_title(f"Augmented {i+1}")
        axes[i].axis('off')
    plt.show()
   
def plot_loss_and_acc(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(all_targets, all_predictions, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = confusion_matrix(all_targets, all_predictions)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    

def plot_feature_map_tSNE(model, test_loader, device):
    # Get feature representation for test data
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to the same device as the model
            data = data.to(device)
            # Assuming your model outputs the feature before the final fully connected layer
            # Chain calls to blocks, considering the block type and MaxPool layers
            x = model.initial_layers(data)
            for block in model.blocks:
                # if isinstance(block, (BlockC)):
                #     x = block(x, test_mode=True)
                # # elif isinstance(block, nn.MaxPool2d):
                # #     x = block(x)
                # else:
                    x = block(x)

            features = model.global_avg_pool(x).reshape(data.size(0), -1)  # Get features from global average pooling
            all_features.extend(features.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Apply t-SNE to reduce the dimensionality to 2
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(np.array(all_features))

    # Plot the results
    plt.figure(figsize=(10, 8))
    for label in range(10):
        indices = [i for i, l in enumerate(all_labels) if l == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(label))

    plt.legend()
    plt.title("t-SNE Visualization of Layer 11 Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    
def visualize_feature_maps(model, image, device):
    model.eval()
    image = image.to(device).unsqueeze(0)

    feature_maps = []
    hooks = []

    def hook_fn(module, input, output):
        """ extracting output layers """
        with torch.no_grad():
            feature_map = torch.max(output[0], dim=0)[0].cpu().numpy()
            feature_maps.append(feature_map)

    for layer in model.blocks:
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, BlockE):
            hooks.append(layer.register_forward_hook(hook_fn))

    _ = model(image)

    for hook in hooks:
        hook.remove()

    num_layers = len(feature_maps)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

    for i, fmap in enumerate(feature_maps):
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Layer {i+1}')

    plt.show()    


def show_random_faces(num_people=2, min_images=3, image_dir = './data/face_dataset/dataset/train'):
    """Shows a random set of faces for each person."""

    people = random.sample(os.listdir(image_dir), num_people)

    for person in people:
        person_dir = os.path.join(image_dir, person)
        if os.path.isdir(person_dir):  # Ensure it's a directory
            image_files = [f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))]

          # if len(image_files) >= min_images:
            img_num = min(min_images, len(image_files))
            selected_images = random.sample(image_files, img_num)

            plt.figure(figsize=(10, 5))
            for i, image_file in enumerate(selected_images):
              image_path = os.path.join(person_dir, image_file)
              try:
                  img = Image.open(image_path)
                  plt.subplot(1, img_num, i + 1)
                  plt.imshow(img)
                  plt.title(f"Person: {person}")
                  plt.axis('off')
              except (IOError, OSError) as e:
                  print(f"Error opening image {image_path}: {e}")
        else:
            print(f"{person_dir} is not a directory.")

            plt.show()

def show_image_pairs(train_loader):
    """Prints some image pairs with their labels from the train dataset."""
    for img1, img2, label in train_loader:
        for i in range(min(5, len(img1))):  # Adjust number of pairs displayed
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(img1[i].permute(1, 2, 0))  # Assuming images are in CxHxW format
            plt.title("Image 1")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img2[i].permute(1, 2, 0))
            plt.title("Image 2")
            plt.axis('off')

            plt.suptitle(f"Label: {label[i].item()}")
            plt.show()
        break  # Stop after showing the first batch

def plot_loss_curve(train_losses, val_losses, loss):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({loss})")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

def analyze_feature_distances(model, dataloader, device):
    """
    Calculates and plots the feature distances for same and different person image pairs.
    """
    model.eval()
    distances_same = []
    distances_diff = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            emb1, emb2 = model(img1, img2)
            distances = torch.norm(emb1 - emb2, p=2, dim=1).cpu().numpy()

            for i in range(len(label)):
                if label[i].item() == 1:
                    distances_same.append(distances[i])
                else:
                    distances_diff.append(distances[i])

    # Create box plots
    plt.figure(figsize=(8, 6))
    plt.boxplot([distances_same, distances_diff], tick_labels=['Same Person', 'Different Persons'])
    plt.xlabel('Image Pairs')
    plt.ylabel('Feature Distance')
    plt.title('Feature Distance Box Plots')
    plt.show()

    # Stacked Histogram
    plt.figure(figsize=(8, 6))
    plt.hist([distances_same, distances_diff], bins=30, stacked=True, label=['Same Person', 'Different Person'], alpha=0.5)
    plt.xlabel('Feature Distance')
    plt.ylabel('Number of Image Pairs')
    plt.title('Feature Distance Histogram')
    plt.legend()
    plt.show()

def plot_k_nearest_neighbors(model, dataloader, device, num_images = 5, k=10):
    """
    Plots k-nearest neighbors for 5 random images from the dataset.
    """
    model.eval()
    image_features = []
    image_labels = []
    all_images = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1, emb2 = model(img1, img2)
            image_features.extend(emb1.cpu().numpy())
            image_features.extend(emb2.cpu().numpy())
            all_images.extend(img1.cpu()) # Move images to CPU here
            all_images.extend(img2.cpu()) # Move images to CPU here
            image_labels.extend(label)
            image_labels.extend(label)


    image_features = np.array(image_features)

    # Select random images
    random_indices = np.random.choice(len(image_features), num_images, replace=False)

    for index in random_indices:
        # Calculate cosine similarity
        similarities = cosine_similarity([image_features[index]], image_features)

        # Get indices of k-nearest neighbors (excluding the image itself)
        nearest_neighbors_indices = np.argsort(similarities[0])[::-1][1:k+1]

        # Plot the original image and its nearest neighbors
        plt.figure(figsize=(15, 4))

        # Original image
        plt.subplot(1, k + 1, 1)
        plt.imshow(all_images[index].permute(1, 2, 0).numpy()) # Convert to numpy array
        plt.title(f"(Idx: {index})")
        plt.axis('off')

        # Nearest neighbors
        for i, neighbor_index in enumerate(nearest_neighbors_indices):
            plt.subplot(1, k + 1, i + 2)
            plt.imshow(all_images[neighbor_index].permute(1, 2, 0).numpy()) # Convert to numpy array
            plt.title(f"Neighbor {i + 1}")
            plt.axis('off')

        plt.show()
        
                