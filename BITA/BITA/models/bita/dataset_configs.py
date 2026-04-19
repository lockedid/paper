"""
Dataset-specific configuration for adaptive training strategy
"""

DATASET_CONFIGS = {
    'rsicd': {
        'gate_init_bias': -1.0,      # Increased from -1.5 to allow more agent utilization
        'agent_num': 64,             # Restored to 64 (48 was too small, causing information bottleneck)
        'scale_weights_init': [0.5, 0.3, 0.2],  # Balanced multi-scale
        'learning_rate': 5e-5,
        'weight_decay': 0.05,
        'max_epoch': 15
    },
    'nwpu': {
        'gate_init_bias': -1.0,      # Increased from -1.5
        'agent_num': 64,             # Restored to 64
        'scale_weights_init': [0.5, 0.3, 0.2],  # Unified multi-scale
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'max_epoch': 10
    },
    'ucm': {
        'gate_init_bias': -1.0,      # Increased from -1.5 (was 0.0 before)
        'agent_num': 64,             # Restored to 64
        'scale_weights_init': [0.5, 0.3, 0.2],  # Unified multi-scale
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'max_epoch': 10
    }
}

def get_dataset_config(dataset_name):
    """Get configuration for specific dataset"""
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]
    else:
        # Default configuration - use unified settings
        return DATASET_CONFIGS['rsicd']
