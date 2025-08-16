import tarfile
import zipfile
import os
import random
from collections import defaultdict
import pickle
from logging import raiseExceptions
import subprocess
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from utils.visualization import *

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        if self.train:
            data_dir = os.path.join(self.root_dir, 'cifar-10-batches-py')
            for i in range(1, 6):
                batch_path = os.path.join(data_dir, f'data_batch_{i}')
                with open(batch_path, 'rb') as f:
                    entry = pickle.load(f, encoding='bytes')
                    self.data.extend(entry[b'data'])
                    self.targets.extend(entry[b'labels'])
        else:
            data_dir = os.path.join(self.root_dir, 'cifar-10-batches-py')
            batch_path = os.path.join(data_dir, 'test_batch')
            with open(batch_path, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                self.data.extend(entry[b'data'])
                self.targets.extend(entry[b'labels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = img.reshape(3, 32, 32).transpose((1, 2, 0))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

def download_and_extract_cifar10(root_dir='./data/cifar10'):
    """Downloads and extracts the CIFAR-10 dataset if not already present."""
    dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dataset_filename = "cifar-10-python.tar.gz"
    extracted_dir = "cifar-10-batches-py"

    if os.path.exists(os.path.join(root_dir, extracted_dir)):
        print("CIFAR-10 dataset already exists. Skipping download.")
        return

    print("Downloading CIFAR-10 dataset...")
    torch.hub.download_url_to_file(dataset_url, dataset_filename)

    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(dataset_filename, "r:gz") as tar:
        tar.extractall(root_dir)

    os.remove(dataset_filename)
    print("CIFAR-10 dataset downloaded and extracted successfully.")
  
def load_cifar10(data_dir='./data/cifar10', batch_size=64):
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124], 
            std=[0.24703233, 0.24348505, 0.26158768]
        )
    ])

    download_and_extract_cifar10(root_dir=data_dir)

    train_dataset = CIFAR10Dataset(root_dir=data_dir, train=True, transform=transform)
    test_dataset = CIFAR10Dataset(root_dir=data_dir, train=False, transform=transform)
    
    plot_random_images(train_dataset)
    
    # Create a dictionary to store indices of each class
    class_indices = defaultdict(list)
    for i, (image, label) in enumerate(train_dataset):
        class_indices[label].append(i)

    # Create a list to store indices for validation set
    val_indices = []
    for label in range(10):
        val_indices.extend(class_indices[label][:1000])  # Select 1000 instances from each class

    # Create validation and training datasets using Subset
    val_dataset = Subset(train_dataset, val_indices)
    train_indices = [i for i in range(len(train_dataset)) if i not in val_indices]
    train_dataset = Subset(train_dataset, train_indices)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    valdata_target_distribution(val_dataset)
    
    plot_augmentation_results(train_dataset)

    return train_loader, val_loader, test_loader

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_split=0.001, seed=42):
        """
        Args:
            root_dir (str): Directory with subdirectories for each person's images.
            transform (callable, optional): Transformations to apply to the images.
            split (str): One of ['train', 'val', 'test'].
            val_split (float): Fraction of training data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.split = split
        random.seed(seed)

        # Get list of people (subdirectories)
        self.people = os.listdir(root_dir)

        # Create a dictionary mapping person -> images
        self.data = {}
        for person in self.people:
            person_path = os.path.join(root_dir, person)
            if os.path.isdir(person_path):
                images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
                if len(images) >= 2:  # Ensure at least two images per person
                    self.data[person] = images

        # Split validation data from training
        if split in ['train', 'val']:
            self._split_validation(val_split, seed)

        # Create list of pairs for sampling
        self.pairs = self._generate_pairs()

    def _split_validation(self, val_split, seed):
        """Splits training data into train and validation sets, ensuring each person has at least two images per split."""
        random.seed(seed)
        self.train_data = {}
        self.val_data = {}

        for person, images in self.data.items():
            random.shuffle(images)
            split_idx = max(2, int(len(images) * (1 - val_split)))  # Ensure at least two images per set
            self.train_data[person] = images[:split_idx]
            self.val_data[person] = images[split_idx:] if len(images[split_idx:]) >= 2 else images[:2]

        self.data = self.train_data if self.split == 'train' else self.val_data
        self.people = list(self.data.keys())

    def _generate_pairs(self):
        """Generate pairs of images dynamically based on a 50% chance rule."""
        pairs = []
        for person in self.people:
            images = self.data[person]
            num_images = len(images)

            for img1 in images:
                if random.random() < 0.5:  # 50% chance to pick same person
                    img2 = random.choice([img for img in images if img != img1])
                    label = 1
                else:  # 50% chance to pick different person
                    other_person = random.choice([p for p in self.people if p != person])
                    img2 = random.choice(self.data[other_person])
                    label = 0

                pairs.append((img1, img2, label))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def download_kaggle_dataset():
    # Move kaggle.json if not already in place
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        if os.path.exists("kaggle.json"):
            os.rename("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
        else:
            raise FileNotFoundError("kaggle.json not found! Place it in the script directory.")

    # Set permissions and download dataset
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    subprocess.run("kaggle datasets download -d sudhanshu2198/face-recognition-dataset-siamese-network", shell=True)

    # Extract dataset
    zip_file = "face-recognition-dataset-siamese-network.zip"
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("data/face_dataset")
        os.remove(zip_file)
        print("Download and extraction complete!")

def load_kaggle_dataset():
    download_kaggle_dataset()
    
    show_random_faces(num_people=2, min_images=3)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = FaceRecognitionDataset(root_dir="./data/face_dataset/dataset/train", transform=transform, split='train')
    val_dataset = FaceRecognitionDataset(root_dir="./data/face_dataset/dataset/train", transform=transform, split='val')
    test_dataset = FaceRecognitionDataset(root_dir="./data/face_dataset/dataset/val", transform=transform, split='val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Train dataset shape:", len(train_dataset))
    print("Validation dataset shape:", len(val_dataset))
    print("Test dataset shape:", len(test_dataset))
    
    show_image_pairs(train_loader)
    
    return train_loader, val_loader, test_loader
    

