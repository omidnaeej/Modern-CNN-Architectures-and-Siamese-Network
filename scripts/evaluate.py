import torch
from models.model import *
from utils.metrics import *
from utils.visualization import *

def evaluate_model(model, test_loader, class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Evaluation loop for the test data
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    plot_confusion_matrix(all_targets, all_predictions, classes=class_names, normalize=False, title='Confusion Matrix')
    
       
    