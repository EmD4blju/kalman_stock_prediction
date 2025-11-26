"""Custom Kedro dataset for PyTorch model serialization."""

from pathlib import Path
from typing import Any, Dict
import torch
from kedro.io import AbstractDataset


class PyTorchModelDataset(AbstractDataset):
    """Dataset for saving and loading PyTorch models."""
    
    def __init__(self, filepath: str):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path where the model will be saved/loaded.
        """
        self._filepath = Path(filepath)
    
    def _save(self, model: torch.nn.Module) -> None:
        """
        Save PyTorch model to disk.
        
        Args:
            model: PyTorch model to save.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both state_dict and model metadata
        save_dict = {
            'state_dict': model.state_dict(),
            'model_config': {
                'id': model.id,
                'ticker': model.ticker,
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'layer_dim': model.layer_dim,
                'output_dim': model.output_dim
            }
        }
        torch.save(save_dict, self._filepath)
    
    def _load(self) -> torch.nn.Module:
        """
        Load PyTorch model from disk.
        
        Returns:
            Initialized PyTorch model with loaded weights.
        """
        from .lstm_model import LSTMStockModel
        
        model_data = torch.load(self._filepath)
        config = model_data['model_config']
        
        # Initialize model with saved configuration
        model = LSTMStockModel(
            id=config['id'],
            ticker=config['ticker'],
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            layer_dim=config['layer_dim'],
            output_dim=config['output_dim']
        )
        
        # Load trained weights
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        
        return model
    
    def _describe(self) -> Dict[str, Any]:
        """
        Describe the dataset.
        
        Returns:
            Dictionary with dataset description.
        """
        return dict(filepath=str(self._filepath))
