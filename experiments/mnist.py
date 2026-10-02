import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from residual_lo import LinearOperatorLayer, MLP

# ---------------------------------------------------------------------------
# Architecture knobs — edit these; MLP is sized to match the operator params.
# ---------------------------------------------------------------------------
CODES_PER_HEAD = 16
HYPERNET_NEURONS = 1024  # width of each hypernet hidden layer
# ---------------------------------------------------------------------------

DATASETS = {
    "mnist": torchvision.datasets.MNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "kmnist": torchvision.datasets.KMNIST,
}

IN_FEATURES = 28 * 28
NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LinearOperator vs MLP")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="fashionmnist",
        help="28x28 grayscale classification dataset",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="./data")
    return parser.parse_args()


def load_datasets(name: str, data_root: str, batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset_cls = DATASETS[name]
    trainset = dataset_cls(root=data_root, train=True, download=True, transform=transform)
    testset = dataset_cls(root=data_root, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader


def build_models() -> dict[str, torch.nn.Module]:
    linear_operator = LinearOperatorLayer(
        in_features=IN_FEATURES,
        out_features=NUM_CLASSES,
        codes_per_head=CODES_PER_HEAD,
        output_initialization_scale=1.0,
        hypernet_hidden_sizes=[HYPERNET_NEURONS, HYPERNET_NEURONS],
        affine_mode="full",
        generator_type="mlp",
    )
    parameter_budget = linear_operator.count_parameters()
    mlp = MLP.from_parameter_budget(
        in_features=IN_FEATURES,
        out_features=NUM_CLASSES,
        parameter_budget=parameter_budget,
    )
    print(
        f"codes_per_head={CODES_PER_HEAD}  hypernet_neurons={HYPERNET_NEURONS}\n"
        f"linear_operator_params={parameter_budget:,}  "
        f"mlp_params={mlp.count_parameters():,}  "
        f"mlp_hidden={mlp.hidden_sizes}"
    )
    return {
        "linear_operator": linear_operator,
        "mlp": mlp,
    }


def forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, IN_FEATURES).to(device)
            labels = labels.to(device)
            predicted = forward_logits(model, images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def train_model(
    name: str,
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n=== {name} ({param_count:,} parameters) ===")

    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        progress = tqdm(trainloader, desc=f"{name} epoch {epoch + 1}/{epochs}", leave=True)
        for images, labels in progress:
            images = images.view(-1, IN_FEATURES).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(forward_logits(model, images), labels)
            loss.backward()
            optimizer.step()

            steps += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

        train_acc = accuracy(model, trainloader, device)
        test_acc = accuracy(model, testloader, device)
        print(
            f"{name} epoch {epoch + 1}/{epochs}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dataset={args.dataset}  device={device}")

    trainloader, testloader = load_datasets(
        args.dataset, args.data_root, args.batch_size
    )
    models = build_models()

    for name, model in models.items():
        train_model(
            name=name,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
