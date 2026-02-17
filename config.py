"""
Configuration file for Medical Image Classification with GOHBO-Optimized ResNet-18
"""

import os
from pathlib import Path

# ==================== PATH CONFIGURATION ====================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
# Phase 1 output directory for storing optimization and evaluation results
PHASE1_OUTPUT = RESULTS_DIR / 'phase1'
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PHASE1_OUTPUT]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== DATASET CONFIGURATION ====================
DATASET_CONFIGS = {
    'brain_tumor': {
        'name': 'Brain Tumor MRI Classification',
        'num_classes': 4,
        'classes': ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
        'image_size': (224, 224),
        'channels': 3,
        'mean': [0.485, 0.456, 0.406],  # ImageNet statistics
        'std': [0.229, 0.224, 0.225],
        'data_path': DATA_DIR / 'brain_tumor' / 'archive',
        'kaggle_dataset': 'sartajbhuvaji/brain-tumor-classification-mri',
    },
    'chest_xray': {
        'name': 'Chest X-Ray Pneumonia Detection',
        'num_classes': 2,
        'classes': ['NORMAL', 'PNEUMONIA'],
        'image_size': (224, 224),
        'channels': 3,  # Convert grayscale to RGB
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'data_path': DATA_DIR / 'chest_xray',
        'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
    },
    'colorectal': {
        'name': 'Colorectal Cancer Histopathology',
        'num_classes': 8,
        'classes': ['TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY'],
        'image_size': (224, 224),
        'channels': 3,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'data_path': DATA_DIR / 'colorectal',
        'kaggle_dataset': 'kmader/colorectal-histology-mnist',
    }
}

# Default dataset
DEFAULT_DATASET = 'brain_tumor'

# ==================== GOHBO ALGORITHM CONFIGURATION ====================
GOHBO_CONFIG = {
    # General GOHBO parameters
    'population_size': 20,  # Number of candidate solutions
    # D-1: GOHBO runs 50-100 iterations (balanced approach)
    # Maximum number of times GOHBO will try different hyperparameter combinations
    # before picking the best one. More iterations = better chance of finding optimal
    # values, but takes longer to run.
    'max_iterations': 50,   # Maximum optimization iterations
    'tolerance': 1e-6,      # Convergence tolerance
    'seed': 42,            # Random seed for reproducibility

    # D-3: Stop optimization early when hitting 95% accuracy
    # If the model achieves 95% accuracy or higher during optimization,
    # we can stop searching because we've reached our goal!
    # This saves time - no need to keep searching if we already hit the target.
    'target_accuracy': 0.95,  # Target accuracy to trigger early stopping

    # D-2: Optimize core hyperparameters only
    # These are the three most important settings that affect model performance:
    # 1. learning_rate: How fast the model learns (like steps when walking - too big you overshoot, too small takes forever)
    # 2. batch_size: How many images to look at before updating the model (16, 32, or 64)
    # 3. epochs: How many times to go through ALL training images
    'hyperparameters_to_optimize': ['learning_rate', 'batch_size', 'epochs'],

    # Grey Wolf Optimizer (GWO) parameters
    'gwo': {
        'a_initial': 2.0,   # Initial value of 'a' parameter
        'a_final': 0.0,     # Final value of 'a' parameter
        'alpha_weight': 0.5,  # Weight for alpha wolf influence
        'beta_weight': 0.3,   # Weight for beta wolf influence
        'delta_weight': 0.2,  # Weight for delta wolf influence
    },

    # Heap-Based Optimization (HBO) parameters
    'hbo': {
        'heap_size': 10,           # Size of the heap structure
        'replacement_rate': 0.2,   # Rate of solution replacement
        'local_search_prob': 0.3,  # Probability of local search
        'neighborhood_size': 5,    # Size of local neighborhood
    },

    # Orthogonal Learning parameters
    'orthogonal': {
        'factor': 0.3,           # Orthogonal learning strength
        'num_factors': 3,        # Number of orthogonal factors
        'exploration_rate': 0.4,  # Exploration vs exploitation balance
    },

    # Optimization bounds for learning rate
    # Learning rate determines how much the model changes with each update
    # Too high = model learns too fast and overshoots the correct answer
    # Too low = model learns too slowly and takes forever
    # We search between 0.00001 and 0.1 using logarithmic scale
    'learning_rate_bounds': {
        'min': 1e-5,
        'max': 1e-1,
        'scale': 'log',  # Use logarithmic scale for learning rate
    },

    # Batch size bounds
    # Batch size is how many images the model looks at before updating its weights
    # Smaller batches (16) = more updates but noisier learning
    # Larger batches (64) = fewer updates but smoother learning
    # We only try three specific values: 16, 32, or 64
    'batch_size_bounds': {
        'min': 16,
        'max': 64,
        'options': [16, 32, 64],  # Only these specific values are allowed
    },

    # Epochs bounds
    # An epoch is one complete pass through all training images
    # Too few epochs = model doesn't learn enough
    # Too many epochs = model memorizes training data (overfitting)
    # We search between 30 and 150 epochs to find the sweet spot
    'epochs_bounds': {
        'min': 30,
        'max': 150,
    }
}

# ==================== TRAINING CONFIGURATION ====================
TRAINING_CONFIG = {
    # Basic training parameters
    'batch_size': 32,
    'epochs': 100,
    'num_workers': 4,  # Number of data loading workers
    'pin_memory': True,  # Pin memory for faster GPU transfer

    # Optimizer configuration (will use GOHBO-optimized learning rate)
    'optimizer': {
        'type': 'Adam',
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    },

    # Learning rate scheduler
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 50,
        'eta_min': 1e-6,
    },

    # Training strategy
    'gradient_clip_value': 1.0,  # Gradient clipping value
    'gradient_accumulation_steps': 1,  # For larger effective batch size
    'mixed_precision': False,  # Use automatic mixed precision

    # Validation and checkpointing
    'validation_split': 0.15,
    'test_split': 0.15,
    'early_stopping': {
        'enabled': True,
        # D-6: Early stopping patience = 10 epochs
        # This means: if the model does not improve for 10 epochs in a row,
        # we stop training to save time and prevent overfitting.
        # Overfitting = when the model memorizes training data instead of learning patterns.
        # We use 10 instead of 15 because medical images can be noisy, so we need
        # to give the model enough chances to improve, but not waste time if it's stuck.
        'patience': 10,
        'min_delta': 0.001,
        'mode': 'min',  # min for loss, max for accuracy
        'metric': 'val_loss',
    },
    'checkpoint': {
        # D-5: Save training checkpoints every 5 epochs
        # A checkpoint is like a "save point" in a video game - it saves the model's
        # current state so we can resume training if something crashes.
        # Every 5 epochs = good balance between safety and not filling up disk space
        'save_interval': 5,  # Save checkpoint every N epochs
        'save_best_only': True,
        'monitor': 'val_accuracy',
        'mode': 'max',
    },
}

# ==================== DATA AUGMENTATION CONFIGURATION ====================
AUGMENTATION_CONFIG = {
    'train': {
        'random_horizontal_flip': {
            'enabled': True,
            'p': 0.5,
        },
        'random_vertical_flip': {
            'enabled': False,  # Not suitable for chest X-rays
            'p': 0.5,
        },
        'random_rotation': {
            'enabled': True,
            'degrees': 15,
        },
        'color_jitter': {
            'enabled': True,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
        },
        'random_affine': {
            'enabled': True,
            'degrees': 10,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1),
        },
        'gaussian_blur': {
            'enabled': True,
            'kernel_size': 3,
            'p': 0.2,
        },
        'random_erasing': {
            'enabled': True,
            'p': 0.2,
            'scale': (0.02, 0.33),
            'ratio': (0.3, 3.3),
        },
    },
    'val': {
        # No augmentation for validation
    },
    'test': {
        # No augmentation for testing
    }
}

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    'architecture': 'resnet18',
    'pretrained': True,  # Use ImageNet pretrained weights
    'freeze_backbone': False,  # Whether to freeze backbone during training
    'dropout_rate': 0.5,  # Dropout rate in classifier head
    'hidden_units': [512, 256],  # Hidden units in classifier head

    # Feature extraction settings
    'feature_extract': {
        'enabled': False,  # If True, only update classifier weights
        'unfreeze_at_epoch': 10,  # Epoch to unfreeze backbone if feature_extract
    }
}

# ==================== LOGGING AND MONITORING ====================
LOGGING_CONFIG = {
    'tensorboard': {
        'enabled': True,
        'log_dir': RESULTS_DIR / 'tensorboard',
        'log_interval': 10,  # Log every N batches
        'log_images': True,
        'log_graph': True,
        'log_histogram': True,
    },
    'console': {
        'enabled': True,
        'log_interval': 20,  # Print every N batches
        'use_tqdm': True,  # Use tqdm progress bar
    },
    'file': {
        'enabled': True,
        'log_dir': RESULTS_DIR / 'logs',
        'log_level': 'INFO',
    },
    'wandb': {
        'enabled': False,  # Weights & Biases integration
        'project': 'medical-image-classification',
        'entity': None,  # Your wandb username
    }
}

# ==================== EVALUATION CONFIGURATION ====================
EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'confusion_matrix',
        'classification_report',
    ],
    'visualization': {
        'plot_confusion_matrix': True,
        'plot_roc_curves': True,
        'plot_class_distribution': True,
        'plot_training_history': True,
        'save_misclassified': True,
        'num_misclassified_to_save': 20,
    },
    'output_dir': RESULTS_DIR / 'evaluation',
}

# ==================== HARDWARE CONFIGURATION ====================
HARDWARE_CONFIG = {
    'device': 'cuda',  # 'cuda' or 'cpu'
    'cuda_device': 0,  # Which GPU to use if multiple available
    'deterministic': True,  # For reproducibility
    'benchmark': False,  # CUDNN benchmark mode
    'seed': 42,  # Random seed
}

# ==================== EXPERIMENT TRACKING ====================
EXPERIMENT_CONFIG = {
    'name': 'gohbo_resnet18_medical',
    'version': 'v1.0',
    'description': 'Medical image classification with GOHBO-optimized ResNet-18',
    'tags': ['medical-imaging', 'gohbo', 'resnet18', 'optimization'],
    'save_config': True,  # Save config with each experiment
}

def get_config(dataset_name=None):
    """
    Get configuration for a specific dataset

    Args:
        dataset_name: Name of the dataset ('brain_tumor', 'chest_xray', 'colorectal')

    Returns:
        Dictionary containing all configuration parameters
    """
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")

    config = {
        'dataset': DATASET_CONFIGS[dataset_name],
        'gohbo': GOHBO_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'model': MODEL_CONFIG,
        'logging': LOGGING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'paths': {
            'base': BASE_DIR,
            'data': DATA_DIR,
            'src': SRC_DIR,
            'models': MODELS_DIR,
            'results': RESULTS_DIR,
            'notebooks': NOTEBOOKS_DIR,
        }
    }

    return config

if __name__ == '__main__':
    # Test configuration loading
    config = get_config('brain_tumor')
    print(f"Configuration loaded for: {config['dataset']['name']}")
    print(f"Number of classes: {config['dataset']['num_classes']}")
    print(f"GOHBO population size: {config['gohbo']['population_size']}")
    print(f"Training epochs: {config['training']['epochs']}")