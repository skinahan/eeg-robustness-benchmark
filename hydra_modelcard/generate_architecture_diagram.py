"""
Generate architecture diagram for HYDRA model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_architecture_diagram(output_path="figures/architecture_diagram.png"):
    """Create a comprehensive architecture diagram for HYDRA."""
    
    # Constants for consistent spacing and sizing
    BLOCK_HEIGHT = 0.6
    BLOCK_WIDTH = 3.0
    BLOCK_CENTER_X = 5.0
    BLOCK_SPACING = 0.5  # Space between blocks
    ARROW_SPACING = 0.3  # Space for arrows between blocks
    
    # Special block dimensions
    MULTISCALE_HEIGHT = 1.0
    MULTISCALE_WIDTH = 6.0
    BIN_HEIGHT = 0.8
    BIN_WIDTH = 2.0
    
    # Colors
    colors = {
        'input': '#E8F4F8',
        'conv': '#4A90E2',
        'pool': '#7ED321',
        'multiscale': '#F5A623',
        'snr': '#BD10E0',
        'recurrent': '#9013FE',
        'attention': '#50E3C2',
        'output': '#B8E986',
        'binning': '#87CEEB'
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 15)  # Extended to ensure all elements are visible
    ax.axis('off')
    
    # Title
    ax.text(BLOCK_CENTER_X, 13.5, 'HYDRA Architecture (BranchedWiredCfC Architecture #4)', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    y_pos = 12.5
    
    def draw_block(x, y, width, height, color, text, fontsize=9, fontweight='normal'):
        """Helper function to draw a standardized block."""
        box = Rectangle((x - width/2, y - height/2), width, height,
                       facecolor=color, edgecolor='black', linewidth=1.5, zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
                fontweight=fontweight, zorder=2)
        return y
    
    def draw_arrow(x, y, dx, dy, zorder=2):
        """Helper function to draw an arrow."""
        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, 
                fc='black', ec='black', zorder=zorder)
    
    def move_down(amount):
        """Helper to move y_pos down and return new position."""
        nonlocal y_pos
        y_pos -= amount
        return y_pos
    
    # Input
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['input'], 'Input\n(B, 22, 1000)', fontsize=10, fontweight='bold')
    move_down(BLOCK_HEIGHT)
    
    # Arrow
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Temporal Conv
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['conv'], 'Temporal Conv2D\nF1=8, kernel=125')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Depthwise Spatial Conv
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['conv'], 'Depthwise Spatial Conv2D\nF2=16')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Pooling
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['pool'], 'AvgPool2D\npool=4')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Dropout
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               'lightgray', 'Dropout\np=0.25')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING+0.2)
    
    # Multi-Scale Block
    ms_box = Rectangle((BLOCK_CENTER_X - MULTISCALE_WIDTH/2, y_pos - MULTISCALE_HEIGHT/2), 
                       MULTISCALE_WIDTH, MULTISCALE_HEIGHT,
                       facecolor=colors['multiscale'], edgecolor='black', linewidth=1.5, zorder=1)
    ax.add_patch(ms_box)
    ax.text(BLOCK_CENTER_X, y_pos + 0.2, 'Multi-Scale Temporal Block', 
            ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)
    ax.text(3, y_pos - 0.1, 'Branch 1:\nk=9, d=1', ha='center', va='center', fontsize=8, zorder=2)
    ax.text(5, y_pos - 0.1, 'Branch 2:\nk=15, d=4', ha='center', va='center', fontsize=8, zorder=2)
    ax.text(7, y_pos - 0.1, 'Branch 3:\nk=31, d=16', ha='center', va='center', fontsize=8, zorder=2)
    move_down(MULTISCALE_HEIGHT)
    
    draw_arrow(BLOCK_CENTER_X, y_pos + MULTISCALE_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(0.3)
    
    # SNR Gate
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['snr'], 'SNR Gate\nAdaptive Suppression')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Temporal Downsampler
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['conv'], 'Temporal Downsampler\nConv1D, stride=2')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Binning
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['binning'], 'Temporal Binning\nbin_len=48, stride=44')
    move_down(BLOCK_HEIGHT)
    
    # Branched Processing arrows
    branch_y = y_pos
    bin_target_y = y_pos - 0.5  # Target y position for bins
    draw_arrow(BLOCK_CENTER_X, branch_y, -1.5, bin_target_y - branch_y)
    draw_arrow(BLOCK_CENTER_X, branch_y, 0, bin_target_y - branch_y)
    draw_arrow(BLOCK_CENTER_X, branch_y, 1.5, bin_target_y - branch_y)
    
    # Parallel CfC Processing
    bin_x_positions = [2, 5, 8]  # X positions for the three bins
    y_pos = bin_target_y  # Set bins at target position
    bin_bottom = y_pos - BIN_HEIGHT/2
    bin_top = y_pos + BIN_HEIGHT/2
    
    for i, x_pos in enumerate(bin_x_positions):
        label = 'Bin 1\nCfC' if i == 0 else ('Bin 2\nCfC' if i == 1 else 'Bin N\nCfC')
        draw_block(x_pos, y_pos, BIN_WIDTH, BIN_HEIGHT, 
                   colors['recurrent'], label, fontsize=8)
    
    # Move down for spacing before fusion (more space between bins and fusion)
    move_down(BLOCK_SPACING * 3)  # Slightly more space
    fusion_y = y_pos  # Fusion box center position
    
    # Merge arrows: point UPWARD from bottom of bins to top of fusion box
    arrow_start_y = bin_bottom
    arrow_end_y = fusion_y + BLOCK_HEIGHT/2
    arrow_dy = arrow_end_y - arrow_start_y
    
    for i, x_pos in enumerate(bin_x_positions):
        dx = BLOCK_CENTER_X - x_pos  # Horizontal distance to center
        draw_arrow(x_pos, arrow_start_y, dx, arrow_dy)
    
    y_pos = fusion_y  # Set to fusion position
    
    # Inter-bin Fusion
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['attention'], 'Inter-Bin Fusion\n(Attention)')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Classification Head
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['output'], 'Classification Head\nLinear (H → 2)')
    move_down(BLOCK_HEIGHT)
    draw_arrow(BLOCK_CENTER_X, y_pos + BLOCK_HEIGHT/2, 0, -ARROW_SPACING)
    move_down(BLOCK_SPACING)
    
    # Output
    draw_block(BLOCK_CENTER_X, y_pos, BLOCK_WIDTH, BLOCK_HEIGHT, 
               colors['output'], 'Output Logits\n(B, 2)', fontsize=10, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], label='Input/Output'),
        mpatches.Patch(facecolor=colors['conv'], label='Convolution'),
        mpatches.Patch(facecolor=colors['pool'], label='Pooling'),
        mpatches.Patch(facecolor=colors['multiscale'], label='Multi-Scale'),
        mpatches.Patch(facecolor=colors['snr'], label='SNR Gate'),
        mpatches.Patch(facecolor=colors['recurrent'], label='Recurrent (CfC)'),
        mpatches.Patch(facecolor=colors['attention'], label='Attention'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    create_architecture_diagram()

