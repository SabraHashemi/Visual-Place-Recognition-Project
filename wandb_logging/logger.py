"""
Simple and modular Weights & Biases logger for VPR experiments.
"""

import os
from typing import Optional, Dict, Any
import wandb


class WandBLogger:
    """
    A simple wrapper around wandb for logging VPR experiments.
    Handles initialization, config logging, and metrics tracking.
    """
    
    def __init__(
        self,
        project_name: str = "vpr-evaluation",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        **wandb_kwargs
    ):
        """
        Initialize wandb logger.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of this experiment run (auto-generated if None)
            config: Dictionary of hyperparameters/config to log
            enabled: Whether to enable wandb logging (useful for debugging)
            **wandb_kwargs: Additional arguments to pass to wandb.init()
        """
        self.enabled = enabled and (os.getenv("WANDB_DISABLED", "false").lower() != "true")
        
        if not self.enabled:
            return
            
        # Initialize wandb
        init_kwargs = {
            "project": project_name,
            "name": experiment_name,
            "config": config or {},
            **wandb_kwargs
        }
        wandb.init(**init_kwargs)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (for time series)
        """
        if not self.enabled:
            return
            
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Update and log configuration.
        
        Args:
            config: Dictionary of config values to add/update
        """
        if not self.enabled:
            return
            
        wandb.config.update(config)
    
    def finish(self):
        """Finish the wandb run."""
        if self.enabled:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically finishes wandb."""
        self.finish()







