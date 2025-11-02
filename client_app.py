import random
from logging import INFO, WARNING
from typing import Dict, Any
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log
from torch.utils.data import DataLoader
from task import Net, get_transforms, load_data
from task import train as train_fn
from task import test as test_fn
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from task import NORMALIZATION_CONSTANTS

app = ClientApp()

CLASSIFICATION_HEAD_NAME = "classification-head"

# Saves the weights of the classification head into the client's state
def _save_classification_head_to_state(state: RecordDict, net: Net) -> None:
    state[CLASSIFICATION_HEAD_NAME] = ArrayRecord(net.fc2.state_dict())

# Loads the weights of the classification head from the client's state
def _load_classification_head_from_state(state: RecordDict, net: Net) -> None:
    if CLASSIFICATION_HEAD_NAME not in state:
        return
    state_dict = state[CLASSIFICATION_HEAD_NAME].to_torch_state_dict()
    net.fc2.load_state_dict(state_dict, strict=True)

# Checks if the client is malicious and applies the corresponding attack.
# Returns the (potentially modified) DataLoader and a boolean indicating
# if training should be performed.
def _handle_malicious_behavior(
    partition_id: int, run_config: Dict[str, Any], trainloader: DataLoader
) -> tuple[DataLoader, bool]:
    is_malicious = partition_id in run_config.get("malicious-clients-ids", [])
    attack_name = run_config.get("attack_name", "none") if is_malicious else "none"

    if attack_name == "label_flipping":
        log(WARNING, f"CLIENT {partition_id}: Acting as a Label Flipping attacker.")
        return _apply_label_flipping(run_config, trainloader), True
    
    if attack_name == "backdoor":
        log(WARNING, f"CLIENT {partition_id}: Acting as a Backdoor attacker.")
        return _apply_backdoor_attack(run_config, trainloader), True
    
    if attack_name == "byzantine":
        log(WARNING, f"CLIENT {partition_id}: Acting as a Byzantine node. Skipping training.")
        return trainloader, False  # Skip training

    # Honest client
    return trainloader, True

# Applies the Label Flipping attack to the training data
def _apply_label_flipping(
    run_config: Dict[str, Any], trainloader: DataLoader
) -> DataLoader:
    flip_pairs = [[0, 2], [1, 2], [4, 2], [5, 2], [8, 2], [6, 2], [3,2], [7, 2]]  # Example: flip 0->2, 1->2, etc.
    flip_fraction = run_config.get("attack_flip-fraction", 1.0)
    dataset_name = run_config["dataset-name"]
    batch_size = run_config["batch-size"]

    flip_map = {source: target for source, target in flip_pairs}
    
    log(WARNING, f"CLIENT: Acting as Multi-Label Flipping attacker. Map: {flip_map}")

    raw_dataset = trainloader.dataset.with_format(None)

    def transform_with_flip(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Apply normal image transformations
        transforms = get_transforms(dataset_name, is_train=True)
        batch["image"] = [transforms(img) for img in batch["image"]]

        # Flip labels according to the flip_map with the specified fraction
        original_labels = list(batch["label"])
        flipped_labels = []
        num_flipped = 0
        for label in original_labels:
           # Decide whether to flip this label
            if label in flip_map and random.random() < flip_fraction:
                flipped_labels.append(flip_map[label]) 
                num_flipped += 1
            else:
                flipped_labels.append(label)
        
        batch["label"] = flipped_labels
        return batch

    raw_dataset.set_transform(transform_with_flip)

    return DataLoader(raw_dataset, batch_size=batch_size, shuffle=True)

def _apply_backdoor_attack(
    run_config: Dict[str, Any], trainloader: DataLoader
) -> DataLoader:
    # Backdoor attack parameters
    injection_rate = run_config.get("attack_injection_rate", 0.1)
    target_class = run_config.get("attack_target_class", 1)
    
    trigger_value = 1.0 
    
    dataset_name = run_config["dataset-name"]
    batch_size = run_config["batch-size"]

    log(WARNING, f"CLIENT: Acting as Backdoor attacker (Stealthy). Injecting trigger (target: {target_class}) "
                 f"at rate: {injection_rate}")

    raw_dataset = trainloader.dataset.with_format(None)
    
    norm_constants = NORMALIZATION_CONSTANTS.get(dataset_name)
    
    tensor_augmentations = Compose([
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
    ])
    
    tensor_normalization = Normalize(*norm_constants)
    
    def transform_with_backdoor(batch: Dict[str, Any]) -> Dict[str, Any]:
        
        original_images_pil = batch["image"]
        original_labels = list(batch["label"])
        
        poisoned_images_tensor = []
        poisoned_labels = []

        for i in range(len(original_images_pil)):
            
            img_tensor = ToTensor()(original_images_pil[i])
            img_tensor = tensor_augmentations(img_tensor)
            if random.random() < injection_rate:
                img_tensor[:, 0:3, 0:3] = trigger_value
                poisoned_labels.append(target_class)
            else:
                poisoned_labels.append(original_labels[i])

            img_tensor = tensor_normalization(img_tensor)
            
            poisoned_images_tensor.append(img_tensor)
        
        batch["image"] = torch.stack(poisoned_images_tensor)
        batch["label"] = poisoned_labels
        return batch

    raw_dataset.set_transform(transform_with_backdoor)

    return DataLoader(raw_dataset, batch_size=batch_size, shuffle=True)

# Tests the model for Backdoor Attack Success Rate (ASR) 
# on the validation set transformed with the trigger
def _test_backdoor_asr(
    net: Net, 
    valloader: DataLoader, 
    run_config: Dict[str, Any], 
    device: str
) -> float:
    
    target_class = run_config.get("attack_target_class", 1)
    
    trigger_value = 1.0 
    
    dataset_name = run_config["dataset-name"]
    batch_size = run_config["batch-size"]

    raw_dataset = valloader.dataset.with_format(None)

    norm_constants = NORMALIZATION_CONSTANTS.get(dataset_name)
    
    def transform_with_trigger(batch: Dict[str, Any]) -> Dict[str, Any]:
        
        images_pil = batch["image"]
        original_labels = batch["label"]
        
        triggered_images_tensor = []

        for i in range(len(images_pil)):
            img_tensor = ToTensor()(images_pil[i])
            img_tensor[:, 0:3, 0:3] = trigger_value
            img_tensor = Normalize(*norm_constants)(img_tensor)
            triggered_images_tensor.append(img_tensor)

        batch["image"] = torch.stack(triggered_images_tensor)
        batch["label"] = original_labels 
        return batch

    raw_dataset.set_transform(transform_with_trigger)
    triggered_valloader = DataLoader(raw_dataset, batch_size=batch_size)

    net.to(device)
    net.eval()
    
    correct_misclassifications = 0
    total_non_target_samples = 0
    
    with torch.no_grad():
        for batch in triggered_valloader:
            images = batch["image"].to(device)
            original_labels = batch["label"].to(device) 
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            non_target_mask = (original_labels != target_class)
            total_non_target_samples += non_target_mask.sum().item()
            correct_misclassifications += (
                (predicted == target_class) & non_target_mask
            ).sum().item()

    return (
        correct_misclassifications / total_non_target_samples
        if total_non_target_samples > 0
        else 0.0
    )


# Train function
@app.train()
def train(msg: Message, context: Context) -> Message:
    # Initialize model
    model = Net()

    # Get client-specific context and configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    run_config = msg.content["config"]
    
    num_local_epochs = run_config.get("local-epochs", 1)

    # Handle personalization and device placement
    if run_config.get("personalization", False):
        _load_classification_head_from_state(context.state, model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data for the client
    trainloader, _, _, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=run_config["dataset-name"],
        val_split_percentage=run_config["val-split-percentage"],
        seed=run_config["seed"],
        batch_size=run_config["batch-size"],
        partitioner_name=run_config["partitioner-name"],
        dirichlet_alpha=run_config["dirichlet-alpha"],
    )

    # Handle malicious behavior (if any)
    final_trainloader, perform_training = _handle_malicious_behavior(
        partition_id, run_config, trainloader
    )

    train_loss = 0.0

    if perform_training:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
        log(INFO, f"CLIENT {partition_id}: Using standard SGD optimizer.")
        
        train_loss = train_fn(
            net=model,
            trainloader=final_trainloader,
            epochs=num_local_epochs,
            lr=run_config["lr"],
            device=device,
            proximal_mu=run_config["proximal-mu"],
            momentum=run_config["momentum"],
        )
    else:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)


    if run_config.get("attack_name") == "byzantine" and partition_id in run_config.get("malicious-clients-ids", []):
        std_dev = 0.5 # Standard deviation for Gaussian noise
        noisy_state_dict = {
            key: value + torch.randn_like(value) * std_dev
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(noisy_state_dict)

    
    model_record: ArrayRecord

    if run_config.get("personalization", False):
        _save_classification_head_to_state(context.state, model)
        body_state_dict = {
            k: v for k, v in model.state_dict().items() if not k.startswith("fc2.")
        }
        model_record = ArrayRecord(body_state_dict)
    else:
        # Per FedAvg/FedProx standard, invia il modello completo
        model_record = ArrayRecord(model.state_dict())
        
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # Initialize model from server message
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)

    # Get client-specific context and configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    run_config = msg.content["config"]

    # Handle personalization and device placement
    if run_config.get("personalization", False):
        _load_classification_head_from_state(context.state, model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load validation data
    _, valloader, _, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=run_config["dataset-name"],
        val_split_percentage=run_config["val-split-percentage"],
        seed=run_config["seed"],
        batch_size=run_config["batch-size"],
        partitioner_name=run_config["partitioner-name"],
        dirichlet_alpha=run_config["dirichlet-alpha"],
    )

    # Evaluate the model
    eval_loss, eval_acc, conf_matrix = test_fn(model, valloader, device)

    # Backdoor Attack Success Rate (ASR) evaluation
    backdoor_asr = 0.0

    if "backdoor" in run_config.get("attack_name", ""):
        backdoor_asr = _test_backdoor_asr(model, valloader, run_config, device)
        log(INFO, f"CLIENT {partition_id}: Backdoor ASR: {backdoor_asr:.4f}")
        
    # Prepare and return the response
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "backdoor_asr": backdoor_asr,
    }
    content = RecordDict({
        "metrics": MetricRecord(metrics),
        "confusion_matrix": ArrayRecord([conf_matrix]),
    })

    return Message(content=content, reply_to=msg)