import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


class VAETrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        """
    Trainer class for training a Variational Autoencoder (VAE).

    Args:
        model: VAE model instance.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data (optional).
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function (e.g., sum of reconstruction and KL divergence loss).
        device: Device to use for training (CPU or GPU).
    """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_val_loss = float('inf')

    def train(self, epoch):
        """
    Trains the VAE for one epoch with optional validation.

    Args:
        epoch: Current epoch number.
    """
        self.model.train()  # Set model to training mode
        train_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}") as train_progress:  # Progress bar for training
            for data in train_progress:
                inputs = data.to(self.device)

                # Forward pass
                outputs, mu, logvar = self.model(inputs)
                loss = self.criterion(outputs, inputs, mu, logvar)

                # Backward pass and update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_progress.set_postfix({"loss": train_loss / len(data)})  # Update progress bar with loss

            self.visualize_sample()

        train_loss = train_loss / len(self.train_loader.dataset)

        # Validation (if validation loader provided)
        if self.val_loader is not None:
            val_loss = self.evaluate(self.val_loader)
            self.best_val_loss = min(self.best_val_loss, val_loss)  # Update best validation loss

        print(f"Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}")
        if self.val_loader is not None:
            print(f"Validation Loss: {val_loss:.4f}, Best Validation Loss: {self.best_val_loss:.4f}")

    def evaluate(self, data_loader):
        """
    Evaluates the model on a given dataloader.

    Args:
        data_loader: PyTorch DataLoader for evaluation data.

    Returns:
        Average loss on the evaluation data.
    """
        self.model.eval()  # Set model to evaluation mode
        eval_loss = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for data in data_loader:
                inputs, targets = data.to(self.device)

                # Forward pass
                outputs, mu, logvar = self.model(inputs)
                loss = self.criterion(outputs[0], targets, mu, logvar)

                eval_loss += loss.item()

        eval_loss = eval_loss / len(data_loader.dataset)
        return eval_loss

    def save_model(self, filepath):
        """
    Saves the trained VAE model to a file.

    Args:
        filepath: Path to save the model file.
    """
        torch.save(self.model.state_dict(), filepath)

    def visualize_sample(self, num_samples: int = 3) -> None:
        samples = self.model.sample(num_samples)
        # Create figure and subplots
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))

        # Loop through channels and plot
        for i, channel in enumerate(samples):
            # Convert channel tensor to numpy array
            channel_img = channel.cpu().numpy()

            # Plot on a subplot
            axs[i].imshow(channel_img, cmap="gray", vmin=0, vmax=1)
            axs[i].set_title(f"Channel {i + 1}")
            axs[i].axis("off")

        # Show the plot
        plt.show()
