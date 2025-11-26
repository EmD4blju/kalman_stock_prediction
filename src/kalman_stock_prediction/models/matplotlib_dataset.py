"""Custom Kedro dataset for saving Matplotlib figures."""

from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
from kedro.io import AbstractDataset


class MatplotlibDataset(AbstractDataset):
    """Dataset for saving Matplotlib figures as image files."""
    
    def __init__(self, filepath: str, save_args: Dict[str, Any] = None):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path where the figure will be saved (e.g., 'path/to/plot.png').
            save_args: Additional arguments to pass to plt.Figure.savefig().
                      Common options: dpi (default: 300), bbox_inches (default: 'tight'),
                      format (inferred from filepath extension).
        """
        self._filepath = Path(filepath)
        self._save_args = save_args or {}
        
        # Set sensible defaults for save_args
        if 'dpi' not in self._save_args:
            self._save_args['dpi'] = 300
        if 'bbox_inches' not in self._save_args:
            self._save_args['bbox_inches'] = 'tight'
    
    def _save(self, figure: plt.Figure) -> None:
        """
        Save Matplotlib figure to disk.
        
        Args:
            figure: Matplotlib figure to save.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self._filepath, **self._save_args)
        plt.close(figure)  # Close the figure to free memory
    
    def _load(self) -> None:
        """
        Loading is not supported for Matplotlib figures.
        
        Raises:
            NotImplementedError: Loading matplotlib figures is not supported.
        """
        raise NotImplementedError(
            "Loading Matplotlib figures is not supported. "
            "This dataset is write-only for saving visualization outputs."
        )
    
    def _describe(self) -> Dict[str, Any]:
        """
        Describe the dataset.
        
        Returns:
            Dictionary with dataset description.
        """
        return dict(
            filepath=str(self._filepath),
            save_args=self._save_args
        )
