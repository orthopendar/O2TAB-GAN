"""
O2TAB-GAN: Orthopaedic Oncology Tabular GAN
Author: Dr. Ehsan Pendar
Date: July 4, 2025
Description: O2TAB-GAN Generator with FT-Transformer and Fourier-feature MLPs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional

try:
    import rtdl
except ImportError:
    print("Warning: rtdl not found. Please install with: pip install rtdl-revisiting-models")


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping to overcome spectral bias in MLPs.
    Maps input through sin/cos of random projections.
    """
    
    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        
        # Random projection matrix (frozen)
        self.register_buffer(
            'B', 
            torch.randn(in_features, out_features // 2) * self.scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Fourier features of shape (batch_size, out_features)
        """
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatureMLP(nn.Module):
    """
    MLP with Fourier feature preprocessing for numerical data.
    Designed to capture complex, non-Gaussian distributions.
    """
    
    def __init__(
        self, 
        num_numerical_features: int, 
        fourier_dim: int = 128, 
        fourier_scale: float = 10.0, 
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        
        if num_numerical_features == 0:
            self.fourier_features = None
            self.mlp = None
            return
            
        # Fourier feature mapping
        self.fourier_features = FourierFeatures(
            num_numerical_features, 
            fourier_dim, 
            fourier_scale
        )
        
        # MLP layers
        layers = []
        prev_dim = fourier_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_numerical_features))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process numerical features through Fourier-feature MLP.
        
        Args:
            x: Input tensor of shape (batch_size, num_numerical_features)
            
        Returns:
            Processed features of same shape
        """
        if self.fourier_features is None or self.mlp is None:
            return x
            
        ff = self.fourier_features(x)
        return self.mlp(ff)


class FTTransformerEncoder(nn.Module):
    """
    FT-Transformer encoder wrapper for categorical features.
    Uses attention mechanism for rich feature representations.
    """
    
    def __init__(
        self, 
        num_numerical_features: int,
        cat_cardinalities: List[int],
        d_token: int = 128,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1
    ):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.cat_cardinalities = cat_cardinalities
        
        if len(cat_cardinalities) == 0 and num_numerical_features == 0:
            self.transformer = None
            self.output_dim = d_token
            return
        
        try:
            self.transformer = rtdl.FTTransformer.make_default(
                n_num_features=num_numerical_features if num_numerical_features > 0 else None,
                cat_cardinalities=cat_cardinalities if len(cat_cardinalities) > 0 else None,
                d_token=d_token,
                n_blocks=n_blocks,
                attention_n_heads=attention_n_heads,
                ffn_d_hidden_multiplier=4.0 / 3.0,
                ffn_dropout=ffn_dropout,
                residual_dropout=0.0,
                attention_dropout=attention_dropout,
                d_out=d_token
            )
            self.output_dim = d_token
        except Exception as e:
            print(f"Warning: FT-Transformer initialization failed: {e}")
            print("Falling back to simple embedding layers")
            
            # Fallback: simple embedding layers
            self.transformer = None
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in cat_cardinalities
            ])
            self.output_dim = d_token * max(1, len(cat_cardinalities))

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Process features through FT-Transformer.
        
        Args:
            x_num: Numerical features of shape (batch_size, num_numerical_features)
            x_cat: Categorical features of shape (batch_size, num_categorical_features)
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        if self.transformer is not None:
            return self.transformer(x_num, x_cat)
        
        # Fallback implementation
        if x_cat is not None and len(self.categorical_embeddings) > 0:
            cat_embeddings = []
            for i, embedding_layer in enumerate(self.categorical_embeddings):
                if i < x_cat.shape[1]:
                    cat_embeddings.append(embedding_layer(x_cat[:, i].long()))
            
            if cat_embeddings:
                return torch.cat(cat_embeddings, dim=-1)
        
        # If no categorical features, return zero tensor
        # Try to get batch size from x_num, x_cat, or use 1 as fallback
        if x_num is not None:
            batch_size = x_num.shape[0]
        elif x_cat is not None:
            batch_size = x_cat.shape[0]
        else:
            batch_size = 1
        
        device = next(self.parameters()).device if hasattr(self, 'categorical_embeddings') and len(self.categorical_embeddings) > 0 else torch.device('cpu')
        return torch.zeros(batch_size, self.output_dim, device=device)


class O2Generator(nn.Module):
    """
    O2TAB-GAN Generator: Hybrid architecture combining FT-Transformer 
    and Fourier-feature MLPs for superior tabular data generation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_info: List,
        num_numerical_features: int = 0,
        cat_cardinalities: Optional[List[int]] = None,
        d_token: int = 128,
        fourier_dim: int = 128,
        fourier_scale: float = 10.0,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        if cat_cardinalities is None:
            cat_cardinalities = []
            
        self.input_dim = input_dim
        self.output_info = output_info
        self.num_numerical_features = num_numerical_features
        self.cat_cardinalities = cat_cardinalities
        
        # Calculate output dimensions
        self.total_output_dim = sum(item[0] for item in output_info)
        
        # Initial MLP to process noise and conditional vectors
        self.initial_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Enhanced encoders for different data types
        self.numerical_encoder = FourierFeatureMLP(
            num_numerical_features=num_numerical_features,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
            hidden_dims=hidden_dims
        )
        
        self.categorical_encoder = FTTransformerEncoder(
            num_numerical_features=0,  # FT-Transformer handles categoricals separately
            cat_cardinalities=cat_cardinalities,
            d_token=d_token
        )
        
        # Final output layer
        combined_features_dim = (
            hidden_dims[-1] + 
            (num_numerical_features if num_numerical_features > 0 else 0) +
            self.categorical_encoder.output_dim
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(combined_features_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], self.total_output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, noise: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate synthetic data from noise and conditional vectors.
        
        Args:
            noise: Random noise tensor of shape (batch_size, noise_dim)
            cond: Conditional vector of shape (batch_size, cond_dim)
            
        Returns:
            Generated data of shape (batch_size, total_output_dim)
        """
        batch_size = noise.shape[0]
        
        # Combine noise and conditional vector
        if cond is not None:
            x = torch.cat([noise, cond], dim=1)
        else:
            x = noise
        
        # Process through initial MLP
        initial_features = self.initial_mlp(x)
        
        # Split initial features for different encoders
        split_point = self.num_numerical_features if self.num_numerical_features > 0 else 0
        
        features_to_combine = [initial_features]
        
        # Process numerical features if present
        if self.num_numerical_features > 0:
            numerical_input = initial_features[:, :split_point]
            enhanced_numerical = self.numerical_encoder(numerical_input)
            features_to_combine.append(enhanced_numerical)
        
        # Process categorical features if present
        if len(self.cat_cardinalities) > 0:
            # Create dummy categorical input with correct batch size
            # In a real implementation, this would be more sophisticated
            cat_input = torch.zeros(batch_size, len(self.cat_cardinalities), 
                                   device=noise.device, dtype=torch.long)
            enhanced_categorical = self.categorical_encoder(None, cat_input)
            features_to_combine.append(enhanced_categorical)
        
        # Combine all features
        combined_features = torch.cat(features_to_combine, dim=-1)
        
        # Generate final output
        output = self.output_layer(combined_features)
        
        return output
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


class O2Discriminator(nn.Module):
    """
    Enhanced discriminator for O2TAB-GAN with improved architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between real and fake data.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Discrimination scores of shape (batch_size, 1)
        """
        return self.model(x) 