from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

# Normalization constants for image datasets
NORMALIZATION_CONSTANTS = {
    "zalando-datasets/fashion_mnist": ((0.1307,), (0.3081,)),
    "ylecun/mnist": ((0.1307,), (0.3081,)),
}

# Model Definition
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        # The final layer can be treated as the personalization layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training Function
def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    proximal_mu: float,
    momentum: float,
) -> float:
    
    net.to(device)
    net.train()

    # Store a copy of the global model parameters for FedProx
    global_params = (
        [p.data.clone() for p in net.parameters()] if proximal_mu > 0 else None
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    total_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Add FedProx proximal term if enabled
            if proximal_mu > 0 and global_params:
                print("Applying FedProx proximal term..")
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += torch.pow(torch.norm(local_param - global_param), 2)
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

    return total_loss / len(trainloader.dataset)

# Testing Function  
def test(
    net: Net, testloader: DataLoader, device: str
) -> Tuple[float, float, np.ndarray]:
    
    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)

            total_loss += criterion(outputs, labels).item() * len(labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    
    # Generate confusion matrix for F1 score calculation
    cm = confusion_matrix(all_labels, all_preds, labels=range(10))
    return avg_loss, accuracy, cm

# Transformations for training and evaluation
def get_transforms(dataset_name: str, is_train: bool) -> Compose:
    norm_constants = NORMALIZATION_CONSTANTS.get(dataset_name)

    if is_train:
        # Augmentations for the training set
        return Compose(
            [
                RandomCrop(28, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*norm_constants),
            ]
        )
    
    # No augmentations for the evaluation set
    return Compose([ToTensor(), Normalize(*norm_constants)])

def _apply_transforms(batch: Dict[str, Any], transforms: Compose) -> Dict[str, Any]:
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

# Global cache for the FederatedDataset to avoid re-downloading
fds: Optional[FederatedDataset] = None

# Load a federated data partition for a client
def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    val_split_percentage: float,
    seed: int,
    batch_size: int,
    partitioner_name: str,
    dirichlet_alpha: float,
) -> Tuple[DataLoader, DataLoader, callable, callable]:

    global fds
    if fds is None:
        # Select the partitioner based on the configuration
        if partitioner_name == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif partitioner_name == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown partitioner name: {partitioner_name}")

        # Download and partition the dataset
        fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

    # Load the specific partition for the client
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=val_split_percentage, seed=42)

    # Create callable transform functions
    train_transforms = get_transforms(dataset_name, is_train=True)
    eval_transforms = get_transforms(dataset_name, is_train=False)

    train_transforms_fn = lambda batch: _apply_transforms(batch, train_transforms)
    eval_transforms_fn = lambda batch: _apply_transforms(batch, eval_transforms)

    # Lazily apply transforms when data is loaded
    train_partition = partition_train_test["train"].with_transform(train_transforms_fn)
    test_partition = partition_train_test["test"].with_transform(eval_transforms_fn)

    # Create DataLoaders
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=batch_size)

    # Return DataLoaders and the raw transform functions
    return trainloader, testloader, train_transforms_fn, eval_transforms_fn

# Create a unique timestamped directory for storing run outputs
def create_run_dir() -> Tuple[Path, str]:
    run_dir_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path.cwd() / "outputs" / run_dir_str
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path, run_dir_str