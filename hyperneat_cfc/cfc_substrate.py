"""
CfC Substrate for HyperNEAT Evolution

This module defines the substrate (geometric layout) for HyperNEAT evolution
of CfC networks. The substrate represents the spatial arrangement of CfC cells
and their connections in a 2D or 3D space.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ncps.wirings import Wiring


@dataclass
class CfCCell:
    """Represents a single CfC cell in the substrate"""
    x: float  # X coordinate in substrate space
    y: float  # Y coordinate in substrate space
    cell_type: str  # 'input', 'hidden', 'output'
    cell_id: int  # Unique identifier
    features: Optional[Dict[str, Any]] = None  # Additional cell properties


class CfCSubstrate:
    """
    Substrate for HyperNEAT evolution of CfC networks.
    
    The substrate defines the geometric layout of CfC cells in 2D space,
    where HyperNEAT can evolve connection patterns between cells.
    """
    
    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 64,
        output_size: int = 2,
        substrate_width: float = 10.0,
        substrate_height: float = 10.0,
        layout_type: str = "grid"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.substrate_width = substrate_width
        self.substrate_height = substrate_height
        self.layout_type = layout_type
        
        # Generate cell layout
        self.cells = self._generate_cell_layout()
        
    def _generate_cell_layout(self) -> List[CfCCell]:
        """Generate the geometric layout of CfC cells"""
        cells = []
        cell_id = 0
        
        if self.layout_type == "grid":
            cells.extend(self._generate_grid_layout(cell_id))
        elif self.layout_type == "circular":
            cells.extend(self._generate_circular_layout(cell_id))
        elif self.layout_type == "hierarchical":
            cells.extend(self._generate_hierarchical_layout(cell_id))
        else:
            raise ValueError(f"Unknown layout type: {self.layout_type}")
            
        return cells
    
    def _generate_grid_layout(self, start_id: int) -> List[CfCCell]:
        """Generate a grid-based layout"""
        cells = []
        cell_id = start_id
        
        # Input layer (bottom row)
        input_spacing = self.substrate_width / (self.input_size + 1)
        for i in range(self.input_size):
            x = (i + 1) * input_spacing
            y = 1.0  # Bottom row
            cells.append(CfCCell(
                x=x, y=y, cell_type='input', cell_id=cell_id,
                features={'channel_idx': i}
            ))
            cell_id += 1
        
        # Hidden layers (middle rows)
        hidden_rows = max(1, self.hidden_size // 16)  # Approximate rows needed
        hidden_per_row = max(1, self.hidden_size // hidden_rows)
        
        for row in range(hidden_rows):
            y = 2.0 + (row + 1) * (self.substrate_height - 4.0) / (hidden_rows + 1)
            row_size = min(hidden_per_row, self.hidden_size - row * hidden_per_row)
            
            for col in range(row_size):
                x = (col + 1) * self.substrate_width / (row_size + 1)
                cells.append(CfCCell(
                    x=x, y=y, cell_type='hidden', cell_id=cell_id,
                    features={'layer_idx': row, 'neuron_idx': col}
                ))
                cell_id += 1
        
        # Output layer (top row)
        output_spacing = self.substrate_width / (self.output_size + 1)
        for i in range(self.output_size):
            x = (i + 1) * output_spacing
            y = self.substrate_height - 1.0  # Top row
            cells.append(CfCCell(
                x=x, y=y, cell_type='output', cell_id=cell_id,
                features={'class_idx': i}
            ))
            cell_id += 1
            
        return cells
    
    def _generate_circular_layout(self, start_id: int) -> List[CfCCell]:
        """Generate a circular layout"""
        cells = []
        cell_id = start_id
        
        # Input layer (outer circle)
        input_radius = 4.0
        for i in range(self.input_size):
            angle = 2 * np.pi * i / self.input_size
            x = self.substrate_width/2 + input_radius * np.cos(angle)
            y = self.substrate_height/2 + input_radius * np.sin(angle)
            cells.append(CfCCell(
                x=x, y=y, cell_type='input', cell_id=cell_id,
                features={'channel_idx': i}
            ))
            cell_id += 1
        
        # Hidden layer (middle circle)
        hidden_radius = 2.0
        for i in range(self.hidden_size):
            angle = 2 * np.pi * i / self.hidden_size
            x = self.substrate_width/2 + hidden_radius * np.cos(angle)
            y = self.substrate_height/2 + hidden_radius * np.sin(angle)
            cells.append(CfCCell(
                x=x, y=y, cell_type='hidden', cell_id=cell_id,
                features={'neuron_idx': i}
            ))
            cell_id += 1
        
        # Output layer (inner circle)
        output_radius = 0.5
        for i in range(self.output_size):
            angle = 2 * np.pi * i / self.output_size
            x = self.substrate_width/2 + output_radius * np.cos(angle)
            y = self.substrate_height/2 + output_radius * np.sin(angle)
            cells.append(CfCCell(
                x=x, y=y, cell_type='output', cell_id=cell_id,
                features={'class_idx': i}
            ))
            cell_id += 1
            
        return cells
    
    def _generate_hierarchical_layout(self, start_id: int) -> List[CfCCell]:
        """Generate a hierarchical layout with multiple layers"""
        cells = []
        cell_id = start_id
        
        # Input layer
        input_y = 1.0
        input_spacing = self.substrate_width / (self.input_size + 1)
        for i in range(self.input_size):
            x = (i + 1) * input_spacing
            cells.append(CfCCell(
                x=x, y=input_y, cell_type='input', cell_id=cell_id,
                features={'channel_idx': i}
            ))
            cell_id += 1
        
        # Multiple hidden layers with decreasing size
        hidden_layers = [self.hidden_size, self.hidden_size//2, self.hidden_size//4]
        for layer_idx, layer_size in enumerate(hidden_layers):
            y = 2.0 + (layer_idx + 1) * (self.substrate_height - 4.0) / (len(hidden_layers) + 1)
            spacing = self.substrate_width / (layer_size + 1)
            
            for i in range(layer_size):
                x = (i + 1) * spacing
                cells.append(CfCCell(
                    x=x, y=y, cell_type='hidden', cell_id=cell_id,
                    features={'layer_idx': layer_idx, 'neuron_idx': i}
                ))
                cell_id += 1
        
        # Output layer
        output_y = self.substrate_height - 1.0
        output_spacing = self.substrate_width / (self.output_size + 1)
        for i in range(self.output_size):
            x = (i + 1) * output_spacing
            cells.append(CfCCell(
                x=x, y=output_y, cell_type='output', cell_id=cell_id,
                features={'class_idx': i}
            ))
            cell_id += 1
            
        return cells
    
    def get_input_cells(self) -> List[CfCCell]:
        """Get all input cells"""
        return [cell for cell in self.cells if cell.cell_type == 'input']
    
    def get_hidden_cells(self) -> List[CfCCell]:
        """Get all hidden cells"""
        return [cell for cell in self.cells if cell.cell_type == 'hidden']
    
    def get_output_cells(self) -> List[CfCCell]:
        """Get all output cells"""
        return [cell for cell in self.cells if cell.cell_type == 'output']
    
    def get_cell_by_id(self, cell_id: int) -> Optional[CfCCell]:
        """Get cell by its ID"""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None
    
    def get_cell_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """Get dictionary mapping cell IDs to coordinates"""
        return {cell.cell_id: (cell.x, cell.y) for cell in self.cells}
    
    def get_substrate_bounds(self) -> Tuple[float, float, float, float]:
        """Get substrate bounds (x_min, x_max, y_min, y_max)"""
        return (0.0, self.substrate_width, 0.0, self.substrate_height)
    
    def visualize_layout(self, save_path: Optional[str] = None):
        """Visualize the substrate layout"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cells by type
        for cell_type, color in [('input', 'blue'), ('hidden', 'green'), ('output', 'red')]:
            type_cells = [cell for cell in self.cells if cell.cell_type == cell_type]
            x_coords = [cell.x for cell in type_cells]
            y_coords = [cell.y for cell in type_cells]
            ax.scatter(x_coords, y_coords, c=color, s=50, alpha=0.7, label=cell_type.capitalize())
        
        ax.set_xlim(0, self.substrate_width)
        ax.set_ylim(0, self.substrate_height)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'CfC Substrate Layout ({self.layout_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 