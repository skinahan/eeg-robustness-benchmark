"""
Generate wiring diagram for Architecture #4 recurrent compartment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import json
from pathlib import Path
import networkx as nx

def load_architecture_4(filepath="outputs/architectures/best_architecture_4_trial_178.json"):
    """Load Architecture #4 wiring configuration."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_wiring_diagram(output_path="figures/wiring_diagram.png", 
                        arch_file="outputs/architectures/best_architecture_4_trial_178.json"):
    """Create a wiring diagram for Architecture #4."""
    
    # Load architecture
    try:
        arch_data = load_architecture_4(arch_file)
    except FileNotFoundError:
        print(f"Warning: Architecture file not found at {arch_file}")
        print("Creating schematic diagram instead...")
        return create_schematic_wiring_diagram(output_path)
    
    input_size = arch_data.get('input_size', 8)
    hidden_size = arch_data.get('hidden_size', 43)
    output_size = arch_data.get('output_size', 7)
    wiring_matrix = np.array(arch_data.get('wiring_matrix', []))
    
    # Create graph representation
    G = nx.DiGraph()
    total_size = input_size + hidden_size + output_size
    
    # Add nodes
    for i in range(input_size):
        G.add_node(i, layer='input')
    for i in range(input_size, input_size + hidden_size):
        G.add_node(i, layer='hidden')
    for i in range(input_size + hidden_size, total_size):
        G.add_node(i, layer='output')
    
    # Add edges based on wiring matrix
    for i in range(total_size):
        for j in range(total_size):
            if abs(wiring_matrix[i][j]) > 1e-6:  # Non-zero connection
                G.add_edge(i, j, weight=wiring_matrix[i][j])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Architecture #4 Recurrent Compartment Wiring', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Subtitle
    ax.text(5, 11, f'Input: {input_size} | Hidden: {hidden_size} | Output: {output_size}', 
            ha='center', va='top', fontsize=12)
    
    # Layout positions
    input_y = 9
    hidden_y = 6
    output_y = 3
    
    # Draw input layer
    input_x_start = 1
    input_spacing = 8 / max(input_size, 1)
    input_nodes = []
    for i in range(input_size):
        x = input_x_start + i * input_spacing
        circle = Circle((x, input_y), 0.15, facecolor='#4A90E2', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        input_nodes.append((x, input_y))
        ax.text(x, input_y-0.4, f'I{i}', ha='center', va='top', fontsize=7)
    
    # Draw hidden layer (sample subset for visualization)
    hidden_x_start = 0.5
    hidden_spacing = 9 / max(min(hidden_size, 20), 1)  # Show up to 20 nodes
    hidden_nodes = []
    nodes_to_show = min(hidden_size, 20)
    for i in range(nodes_to_show):
        x = hidden_x_start + i * hidden_spacing
        circle = Circle((x, hidden_y), 0.12, facecolor='#9013FE', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        hidden_nodes.append((x, hidden_y))
        if i < 5 or i >= nodes_to_show - 2:
            ax.text(x, hidden_y-0.4, f'H{i}', ha='center', va='top', fontsize=7)
    
    if hidden_size > 20:
        ax.text(5, hidden_y-0.4, f'... ({hidden_size} total)', ha='center', va='top', fontsize=8, style='italic')
    
    # Draw output layer
    output_x_start = 2
    output_spacing = 6 / max(output_size, 1)
    output_nodes = []
    for i in range(output_size):
        x = output_x_start + i * output_spacing
        circle = Circle((x, output_y), 0.15, facecolor='#50E3C2', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        output_nodes.append((x, output_y))
        ax.text(x, output_y-0.4, f'O{i}', ha='center', va='top', fontsize=7)
    
    # Draw connections (sample for visualization)
    # Input to Hidden
    input_to_hidden_edges = []
    for i in range(min(input_size, 5)):
        for j in range(min(hidden_size, 10)):
            if i < len(input_nodes) and j < len(hidden_nodes):
                if abs(wiring_matrix[i][input_size + j]) > 1e-6:
                    input_to_hidden_edges.append((i, j))
                    if len(input_to_hidden_edges) < 15:  # Limit connections shown
                        ax.plot([input_nodes[i][0], hidden_nodes[j][0]], 
                               [input_nodes[i][1], hidden_nodes[j][1]], 
                               'b-', alpha=0.3, linewidth=0.5)
    
    # Hidden to Hidden (sample)
    hidden_to_hidden_edges = []
    for i in range(min(hidden_size, 10)):
        for j in range(min(hidden_size, 10)):
            if i != j and i < len(hidden_nodes) and j < len(hidden_nodes):
                if abs(wiring_matrix[input_size + i][input_size + j]) > 1e-6:
                    hidden_to_hidden_edges.append((i, j))
                    if len(hidden_to_hidden_edges) < 20:  # Limit connections shown
                        ax.plot([hidden_nodes[i][0], hidden_nodes[j][0]], 
                               [hidden_nodes[i][1], hidden_nodes[j][1]], 
                               'purple', alpha=0.3, linewidth=0.5)
    
    # Hidden to Output
    hidden_to_output_edges = []
    for i in range(min(hidden_size, 10)):
        for j in range(min(output_size, 5)):
            if i < len(hidden_nodes) and j < len(output_nodes):
                if abs(wiring_matrix[input_size + i][input_size + hidden_size + j]) > 1e-6:
                    hidden_to_output_edges.append((i, j))
                    if len(hidden_to_output_edges) < 15:  # Limit connections shown
                        ax.plot([hidden_nodes[i][0], output_nodes[j][0]], 
                               [hidden_nodes[i][1], output_nodes[j][1]], 
                               'g-', alpha=0.3, linewidth=0.5)
    
    # Layer labels
    ax.text(0.2, input_y, 'Input\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(0.2, hidden_y, 'Hidden\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(0.2, output_y, 'Output\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Statistics box
    stats_text = f"""Wiring Statistics:
• Total Nodes: {total_size}
• Input → Hidden: {len(input_to_hidden_edges)} connections shown
• Hidden → Hidden: {len(hidden_to_hidden_edges)} connections shown
• Hidden → Output: {len(hidden_to_output_edges)} connections shown
• Sparse Connectivity: Yes
• Optimized via Graph Metrics"""
    
    stats_box = FancyBboxPatch((7, 7), 2.5, 3.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow',
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(stats_box)
    ax.text(8.25, 9, stats_text, ha='left', va='top', fontsize=8, family='monospace')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4A90E2', label='Input Units'),
        mpatches.Patch(facecolor='#9013FE', label='Hidden Units'),
        mpatches.Patch(facecolor='#50E3C2', label='Output Units'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label='Input→Hidden'),
        mpatches.Patch(facecolor='none', edgecolor='purple', label='Hidden→Hidden'),
        mpatches.Patch(facecolor='none', edgecolor='green', label='Hidden→Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Wiring diagram saved to {output_path}")
    plt.close()

def create_schematic_wiring_diagram(output_path="figures/wiring_diagram.png"):
    """Create a schematic wiring diagram when architecture file is not available."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Architecture #4 Recurrent Compartment Wiring (Schematic)', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Subtitle
    ax.text(5, 11, 'Input: 8 | Hidden: 43 | Output: 7', 
            ha='center', va='top', fontsize=12)
    
    # Layout positions
    input_y = 9
    hidden_y = 6
    output_y = 3
    
    # Draw input layer
    input_x_start = 1
    input_spacing = 8 / 8
    input_nodes = []
    for i in range(8):
        x = input_x_start + i * input_spacing
        circle = Circle((x, input_y), 0.15, facecolor='#4A90E2', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        input_nodes.append((x, input_y))
        ax.text(x, input_y-0.4, f'I{i}', ha='center', va='top', fontsize=7)
    
    # Draw hidden layer (sample)
    hidden_x_start = 0.5
    hidden_spacing = 9 / 15
    hidden_nodes = []
    for i in range(15):
        x = hidden_x_start + i * hidden_spacing
        circle = Circle((x, hidden_y), 0.12, facecolor='#9013FE', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        hidden_nodes.append((x, hidden_y))
        if i < 3 or i >= 12:
            ax.text(x, hidden_y-0.4, f'H{i}', ha='center', va='top', fontsize=7)
    ax.text(5, hidden_y-0.4, '... (43 total)', ha='center', va='top', fontsize=8, style='italic')
    
    # Draw output layer
    output_x_start = 2
    output_spacing = 6 / 7
    output_nodes = []
    for i in range(7):
        x = output_x_start + i * output_spacing
        circle = Circle((x, output_y), 0.15, facecolor='#50E3C2', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        output_nodes.append((x, output_y))
        ax.text(x, output_y-0.4, f'O{i}', ha='center', va='top', fontsize=7)
    
    # Draw sample connections
    # Input to Hidden
    for i in [0, 2, 4, 6]:
        for j in [0, 3, 6, 9, 12]:
            ax.plot([input_nodes[i][0], hidden_nodes[j][0]], 
                   [input_nodes[i][1], hidden_nodes[j][1]], 
                   'b-', alpha=0.3, linewidth=0.5)
    
    # Hidden to Hidden
    for i in range(0, 12, 2):
        for j in range(i+1, min(i+4, 12)):
            ax.plot([hidden_nodes[i][0], hidden_nodes[j][0]], 
                   [hidden_nodes[i][1], hidden_nodes[j][1]], 
                   'purple', alpha=0.3, linewidth=0.5)
    
    # Hidden to Output
    for i in [0, 3, 6, 9, 12]:
        for j in range(7):
            ax.plot([hidden_nodes[i][0], output_nodes[j][0]], 
                   [hidden_nodes[i][1], output_nodes[j][1]], 
                   'g-', alpha=0.3, linewidth=0.5)
    
    # Layer labels
    ax.text(0.2, input_y, 'Input\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(0.2, hidden_y, 'Hidden\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(0.2, output_y, 'Output\nLayer', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Statistics box
    stats_text = """Wiring Characteristics:
• Sparse Connectivity
• Optimized via Graph Metrics:
  - Entropy
  - Ricci Curvature
  - Algebraic Connectivity
  - Efficiency
• Discovered through Multi-Objective
  Architecture Search"""
    
    stats_box = FancyBboxPatch((7, 7), 2.5, 3.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow',
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(stats_box)
    ax.text(8.25, 9, stats_text, ha='left', va='top', fontsize=8, family='monospace')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4A90E2', label='Input Units (8)'),
        mpatches.Patch(facecolor='#9013FE', label='Hidden Units (43)'),
        mpatches.Patch(facecolor='#50E3C2', label='Output Units (7)'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label='Input→Hidden'),
        mpatches.Patch(facecolor='none', edgecolor='purple', label='Hidden→Hidden'),
        mpatches.Patch(facecolor='none', edgecolor='green', label='Hidden→Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Schematic wiring diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    
    # Try to load actual architecture, fall back to schematic
    arch_file = "outputs/architectures/best_architecture_4_trial_178.json"
    if Path(arch_file).exists():
        create_wiring_diagram(arch_file=arch_file)
    else:
        # Try alternative path
        alt_file = "../outputs/architectures/best_architecture_4_trial_178.json"
        if Path(alt_file).exists():
            create_wiring_diagram(arch_file=alt_file)
        else:
            create_schematic_wiring_diagram()

