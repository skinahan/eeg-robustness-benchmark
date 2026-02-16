"""
Generate compact LaTeX tables in ICML format summarizing default hyperparameters
for EEGNet, REEGNet, CNN-NCPv2, and HYDRA models.

This script extracts default parameters from the factory constructor methods
that wrap each model instance with EEGClassifier.
"""

from pathlib import Path


def format_value(value):
    """Format a value for LaTeX table display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, float):
        if value == 0.0:
            return "0"
        elif abs(value) < 0.001:
            return f"{value:.2e}"
        elif value < 1.0:
            return f"{value:.3f}"
        else:
            return f"{value:.1f}"
    elif isinstance(value, int):
        return str(value)
    elif value is None:
        return "---"
    elif value == "—":
        return "---"
    else:
        return str(value)


def generate_hyperparameter_table():
    """Generate LaTeX table with default hyperparameters for all models."""
    
    # Extract default parameters from factory constructors
    # These are the actual defaults used when creating EEGClassifier instances
    
    models = {
        "EEGNet": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "batch_size": 64,
            "max_epochs": 200,
            "lr_scheduler": "None",
            "gradient_clipping": "None",
            "early_stopping": "Yes",
            "early_stopping_patience": 20,
            "early_stopping_monitor": "valid_loss",
            # Architectural parameters
            "F1": "auto",  # EEGNetv4 handles this internally
            "D": "auto",
            "kernel_length": "auto",
            "drop_prob": "auto",
            "final_conv_length": "auto",
        },
        "REEGNet": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "batch_size": 64,
            "max_epochs": 200,
            "lr_scheduler": "None",
            "gradient_clipping": "None",
            "early_stopping": "Yes",
            "early_stopping_patience": 20,
            "early_stopping_monitor": "valid_loss",
            # Architectural parameters
            "F1": 8,
            "D": 2,
            "kernel_length": 15,
            "drop_prob": 0.15,
            "lstm_hidden_size": 32,
            "lstm_num_layers": 2,
        },
        "CNN-NCPv2": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "batch_size": 32,
            "max_epochs": 200,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_scheduler_patience": 5,
            "lr_scheduler_monitor": "valid_loss",
            "gradient_clipping": "Yes",
            "gradient_clip_value": 1.0,
            "gradient_clip_norm": "L2",
            "early_stopping": "Yes",
            "early_stopping_patience": 20,
            "early_stopping_monitor": "valid_loss",
            # Architectural parameters
            "ncp_hidden_dim": 36,
            "sparsity": 0.7,
        },
        "HYDRA": {
            "optimizer": "AdamW",
            "learning_rate": 1e-2,
            "weight_decay": 0,
            "batch_size": 64,
            "max_epochs": 200,
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_gamma": 0.97,
            "gradient_clipping": "Yes",
            "gradient_clip_value": 1.0,
            "gradient_clip_norm": "L2",
            "early_stopping": "Yes",
            "early_stopping_patience": 20,
            "early_stopping_monitor": "valid_loss",
            # Architectural parameters
            "F1": 8,
            "D": 2,
            "kernel_length": 125,
            "drop_prob": 0.5,
            "temporal_kernel_size": 3,
            "temporal_stride": 2,
            "fusion": "mean",
            # CfC-specific parameters
            "mixed_memory": False,
            "mode": "default",
            "activation": "lecun_tanh",
            "backbone_units": 128,
            "backbone_layers": 1,
            "backbone_dropout": 0.0,
        }
    }
    
    output_lines = []
    
    # Training parameters table
    output_lines.append("% =========================\n")
    output_lines.append("% Training Hyperparameters\n")
    output_lines.append("% =========================\n")
    output_lines.append("\\begin{table}[t]\n")
    output_lines.append("\\caption{\\textbf{Default Training Hyperparameters.} Default training hyperparameters for EEGNet, REEGNet, CNN-NCPv2, and HYDRA models.}\n")
    output_lines.append("\\label{tab:training_hyperparameters}\n")
    output_lines.append("  \\begin{center}\n")
    output_lines.append("    \\begin{small}\n")
    output_lines.append("      \\begin{sc}\n")
    output_lines.append("        \\begin{tabular}{lcccc}\n")
    output_lines.append("          \\toprule\n")
    output_lines.append("          Parameter & EEGNet & REEGNet & CNN-NCPv2 & HYDRA \\\\\n")
    output_lines.append("          \\midrule\n")
    
    # Training parameters rows
    training_params = [
        ("Optimizer", "optimizer"),
        ("Learning Rate", "learning_rate"),
        ("Weight Decay", "weight_decay"),
        ("Batch Size", "batch_size"),
        ("Max Epochs", "max_epochs"),
        ("LR Scheduler", "lr_scheduler"),
        ("LR Scheduler Params", None),  # Special handling
        ("Gradient Clipping", "gradient_clipping"),
        ("Gradient Clip Value", "gradient_clip_value"),
        ("Early Stopping", "early_stopping"),
        ("Early Stopping Patience", "early_stopping_patience"),
    ]
    
    for param_name, param_key in training_params:
        if param_key is None:
            # Special handling for LR scheduler params
            row = [param_name]
            for model_name in ["EEGNet", "REEGNet", "CNN-NCPv2", "HYDRA"]:
                model_data = models[model_name]
                if model_data.get("lr_scheduler") == "None":
                    row.append("---")
                elif model_data.get("lr_scheduler") == "ReduceLROnPlateau":
                    patience = model_data.get("lr_scheduler_patience", "")
                    monitor = model_data.get("lr_scheduler_monitor", "")
                    row.append(f"patience={patience}, monitor={monitor}")
                elif model_data.get("lr_scheduler") == "ExponentialLR":
                    gamma = model_data.get("lr_scheduler_gamma", "")
                    row.append(f"$\\gamma$={gamma}")
                else:
                    row.append("---")
            output_lines.append(f"          {' & '.join(row)} \\\\\n")
        else:
            row = [param_name]
            for model_name in ["EEGNet", "REEGNet", "CNN-NCPv2", "HYDRA"]:
                value = models[model_name].get(param_key, "---")
                if param_key == "gradient_clip_value" and models[model_name].get("gradient_clipping") == "None":
                    row.append("---")
                else:
                    row.append(format_value(value))
            output_lines.append(f"          {' & '.join(row)} \\\\\n")
    
    output_lines.append("          \\bottomrule\n")
    output_lines.append("        \\end{tabular}\n")
    output_lines.append("      \\end{sc}\n")
    output_lines.append("    \\end{small}\n")
    output_lines.append("  \\end{center}\n")
    output_lines.append("  \\vskip -0.1in\n")
    output_lines.append("\\end{table}\n")
    output_lines.append("\n")
    
    # Architectural parameters table
    output_lines.append("% =========================\n")
    output_lines.append("% Architectural Hyperparameters\n")
    output_lines.append("% =========================\n")
    output_lines.append("\\begin{table}[t]\n")
    output_lines.append("\\caption{\\textbf{Default Architectural Hyperparameters.} Default architectural hyperparameters for EEGNet, REEGNet, CNN-NCPv2, and HYDRA models.}\n")
    output_lines.append("\\label{tab:architectural_hyperparameters}\n")
    output_lines.append("  \\begin{center}\n")
    output_lines.append("    \\begin{small}\n")
    output_lines.append("      \\begin{sc}\n")
    output_lines.append("        \\begin{tabular}{lcccc}\n")
    output_lines.append("          \\toprule\n")
    output_lines.append("          Parameter & EEGNet & REEGNet & CNN-NCPv2 & HYDRA \\\\\n")
    output_lines.append("          \\midrule\n")
    
    # Collect all architectural parameters
    arch_params_set = set()
    for model_data in models.values():
        arch_params_set.update([k for k in model_data.keys() if k not in [
            "optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
            "lr_scheduler", "lr_scheduler_patience", "lr_scheduler_monitor", "lr_scheduler_gamma",
            "gradient_clipping", "gradient_clip_value", "gradient_clip_norm",
            "early_stopping", "early_stopping_patience", "early_stopping_monitor"
        ]])
    
    # Sort parameters for consistent ordering
    arch_params_sorted = sorted(arch_params_set)
    
    # Map parameter names to display names
    param_display_names = {
        "F1": "$F_1$ (temporal filters)",
        "D": "$D$ (depth multiplier)",
        "kernel_length": "Kernel length",
        "drop_prob": "Dropout probability",
        "final_conv_length": "Final conv length",
        "lstm_hidden_size": "LSTM hidden size",
        "lstm_num_layers": "LSTM layers",
        "ncp_hidden_dim": "NCP hidden dim",
        "sparsity": "Sparsity",
        "temporal_kernel_size": "Temporal kernel size",
        "temporal_stride": "Temporal stride",
        "fusion": "Fusion type",
        "mixed_memory": "Mixed memory (CfC)",
        "mode": "Mode (CfC)",
        "activation": "Activation (CfC)",
        "backbone_units": "Backbone units (CfC)",
        "backbone_layers": "Backbone layers (CfC)",
        "backbone_dropout": "Backbone dropout (CfC)",
    }
    
    for param_key in arch_params_sorted:
        param_display = param_display_names.get(param_key, param_key.replace("_", " ").title())
        row = [param_display]
        for model_name in ["EEGNet", "REEGNet", "CNN-NCPv2", "HYDRA"]:
            value = models[model_name].get(param_key, "---")
            row.append(format_value(value))
        output_lines.append(f"          {' & '.join(row)} \\\\\n")
    
    output_lines.append("          \\bottomrule\n")
    output_lines.append("        \\end{tabular}\n")
    output_lines.append("      \\end{sc}\n")
    output_lines.append("    \\end{small}\n")
    output_lines.append("  \\end{center}\n")
    output_lines.append("  \\vskip -0.1in\n")
    output_lines.append("\\end{table}\n")
    output_lines.append("\n")
    
    return output_lines


def main():
    """Main function to generate and save LaTeX tables."""
    output_dir = Path(r"E:\Research\Dissertation\full_backup_7_16_2025\moabb_experiments\analysis\analysis\hydra_performance_summary")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    output_lines = generate_hyperparameter_table()
    
    # Write to file
    output_file = output_dir / "hyperparameter_tables.txt"
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"LaTeX hyperparameter tables written to: {output_file}")


if __name__ == "__main__":
    main()
