from __future__ import annotations
"""
Arbitrary wiring module for NCP compatibility with WS-flex graphs.

This module provides a wiring class that can ingest arbitrary WS-flex graphs
from the architecture search outputs and convert them to NCP-compatible wiring.
"""

import numpy as np
import networkx as nx
from ncps.wirings import Wiring
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import logging


from dataclasses import dataclass
from typing import Optional, Union, Literal, Dict, Any
import numpy as np



InputStrategy = Literal["dense", "degree_proportional", "random_io", "degree_weighted_io"]
OutputStrategy = Literal["dense", "uniform", "random_io", "degree_weighted_io"]
EdgeOrientation = Literal["symmetric", "random_oriented", "as_is"]


class WsFlexHiddenWiring(Wiring):
    """
    Build a full (I+H+O) x (I+H+O) wiring from a WS-flex hidden graph.

    The WS-flex graph defines *only* the Hidden <-> Hidden mixing (message exchange).
    Inputs/Outputs live outside the graph and are wired by policy:

      - Inputs  -> Hidden : chosen by `input_strategy`
      - Hidden <-> Hidden : exactly the WS-flex graph (with optional orientation)
      - Hidden -> Outputs : chosen by `output_strategy`

    The resulting full wiring matrix is then handed to your existing ArbitraryWiring.

    Conventions:
      - Adjacency/Wiring matrix W has shape (T, T) where T = I + H + O.
      - W[src, dst] = weight (1.0 by default), i.e., rows are sources, columns are targets.

    Parameters
    ----------
    input_size : int
        Number of input (sensory) units, I.
    hidden_graph : Union[np.ndarray, "nx.Graph", "nx.DiGraph"]
        Hidden-only adjacency (H x H) or a NetworkX (Di)Graph with H nodes (0..H-1).
        If ndarray, non-zeros are treated as edges; values become weights.
    output_size : int
        Number of output (motor) units, O.
    input_strategy : {"dense", "degree_proportional"}
        - "dense": connect every input to every hidden unit.
        - "degree_proportional": each hidden picks a small fan-in of inputs, biased by its graph degree.
    output_strategy : {"dense", "uniform"}
        - "dense": connect every hidden to every output.
        - "uniform": each output receives ~uniform fan-in from hidden units.
    hidden_edge_orientation : {"symmetric","random_oriented","as_is"}
        - "symmetric": for undirected graphs/upper-triangular matrices, add both directions.
        - "random_oriented": pick one direction per undirected edge (deterministic via seed).
        - "as_is": use given directions/weights verbatim.
    add_hidden_self_loops : bool
        If True, add tiny self loops on hidden units when hidden block is empty to ensure minimal recurrence.
    fan_in_inputs : Optional[int]
        For "degree_proportional", number of inputs per hidden (default: max(1, round(log2(I+H)))).
    fan_in_hidden_per_output : Optional[int]
        For "uniform" output strategy, number of hidden per output (default: max(1, round(log2(H)))).
    allow_signed_hidden_edges : bool
        If True and hidden weights lack sign, sample +/- signs using `inhibitory_ratio`.
    inhibitory_ratio : float
        Fraction of inhibitory (negative) hidden edges if `allow_signed_hidden_edges=True`.
    seed : int
        RNG seed for deterministic structured sampling/orientation.

    Notes
    -----
    - Disallowed routes are forcibly zeroed: input->output, output->input, input->input, output->output.
    - If you want sparse I/O, prefer "degree_proportional" + "uniform".
    - This class returns an ArbitraryWiring instance via `.build()`, but you can also
      retrieve the full matrix via `.full_wiring_matrix()`.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_graph: Union[np.ndarray, Any],
                 output_size: int,

                 input_strategy: InputStrategy = "dense",
                 output_strategy: OutputStrategy = "dense",
                 hidden_edge_orientation: EdgeOrientation = "symmetric",
                 add_hidden_self_loops: bool = True,

                 fan_in_inputs: Optional[int] = None,
                 fan_in_hidden_per_output: Optional[int] = None,

                 allow_signed_hidden_edges: bool = False,
                 inhibitory_ratio: float = 0.2,

                 seed: int = 17,
                 dtype: Any = np.float32):
        # Determine hidden size before base init (supports ndarray or networkx)
        try:
            if hasattr(hidden_graph, "shape") and hidden_graph.shape and len(hidden_graph.shape) == 2:
                _hidden_size_init = int(hidden_graph.shape[0])
            elif nx is not None and isinstance(hidden_graph, (nx.Graph, nx.DiGraph)):
                _hidden_size_init = int(len(hidden_graph.nodes()))
            else:
                arr = np.asarray(hidden_graph)
                _hidden_size_init = int(arr.shape[0])
        except Exception:
            arr = np.asarray(hidden_graph)
            _hidden_size_init = int(arr.shape[0])

        super().__init__(units=input_size + _hidden_size_init + output_size)
        self.input_size = input_size
        self.hidden_graph = hidden_graph
        self.output_size = output_size
        self.input_strategy = input_strategy
        self.output_strategy = output_strategy
        self.hidden_edge_orientation = hidden_edge_orientation
        self.add_hidden_self_loops = add_hidden_self_loops
        self.fan_in_inputs = fan_in_inputs
        self.fan_in_hidden_per_output = fan_in_hidden_per_output
        self.allow_signed_hidden_edges = allow_signed_hidden_edges
        self.inhibitory_ratio = inhibitory_ratio
        self.seed = seed
        self.dtype = dtype

    # --------------------------- Public API ---------------------------

    def full_wiring_matrix(self) -> np.ndarray:
        I, H, O = self.input_size, self._hidden_size(), self.output_size
        T = I + H + O
        W = np.zeros((T, T), dtype=self.dtype)

        # 1) Inputs -> Hidden
        if self.input_strategy == "dense":
            W[0:I, I:I + H] = 1.0
        elif self.input_strategy == "degree_proportional":
            rng = np.random.default_rng(self.seed)
            fan_in = self.fan_in_inputs or max(1, int(np.round(np.log2(max(2, I + H)))))
            deg = self._hidden_degrees()
            probs = (deg + 1e-8) / (deg.sum() + 1e-8)
            # Each hidden picks `fan_in` inputs (sampled from inputs uniformly or weighted by hidden degree)
            # We bias *which hidden* is more likely to pick more inputs by sampling hidden order by degree;
            # within a hidden, we choose inputs uniformly without replacement.
            hidden_order = np.argsort(-probs)  # high degree first (deterministic)
            for h in hidden_order:
                chosen_inputs = rng.choice(I, size=min(fan_in, I), replace=False)
                W[chosen_inputs, I + h] = 1.0
        elif self.input_strategy == "random_io":
            rng = np.random.default_rng(self.seed)
            fan_in = self.fan_in_inputs or max(1, int(np.round(np.log2(max(2, I + H)))))
            hidden_order = rng.permutation(H)
            for h in hidden_order:
                chosen_inputs = rng.choice(I, size=min(fan_in, I), replace=False)
                W[chosen_inputs, I + h] = 1.0
        elif self.input_strategy == "degree_weighted_io":
            rng = np.random.default_rng(self.seed)
            fan_in = self.fan_in_inputs or max(1, int(np.round(np.log2(max(2, I + H)))))
            deg = self._hidden_degrees().astype(np.float64)
            probs = (deg + 1e-8) / (deg.sum() + 1e-8)
            hidden_order = rng.choice(H, size=H, replace=False, p=probs)
            for h in hidden_order:
                chosen_inputs = rng.choice(I, size=min(fan_in, I), replace=False)
                W[chosen_inputs, I + h] = 1.0
        else:
            raise ValueError(f"Unknown input_strategy: {self.input_strategy}")

        # 2) Hidden <-> Hidden (WS-flex fabric)
        W[I:I + H, I:I + H] = self._hidden_block_oriented()

        # Ensure minimal recurrence if needed
        if self.add_hidden_self_loops and np.count_nonzero(W[I:I + H, I:I + H]) == 0:
            np.fill_diagonal(W[I:I + H, I:I + H], 0.1)

        # 3) Hidden -> Outputs
        if self.output_strategy == "dense":
            W[I:I + H, I + H:T] = 1.0
        elif self.output_strategy == "uniform":
            rng = np.random.default_rng(self.seed + 1)
            fan_in = self.fan_in_hidden_per_output or max(1, int(np.round(np.log2(max(2, H)))))
            for o in range(O):
                chosen_hidden = rng.choice(H, size=min(fan_in, H), replace=False)
                W[I + chosen_hidden, I + H + o] = 1.0
        elif self.output_strategy == "random_io":
            rng = np.random.default_rng(self.seed + 1)
            fan_in = self.fan_in_hidden_per_output or max(1, int(np.round(np.log2(max(2, H)))))
            for o in range(O):
                chosen_hidden = rng.choice(H, size=min(fan_in, H), replace=False)
                W[I + chosen_hidden, I + H + o] = 1.0
        elif self.output_strategy == "degree_weighted_io":
            rng = np.random.default_rng(self.seed + 1)
            fan_in = self.fan_in_hidden_per_output or max(1, int(np.round(np.log2(max(2, H)))))
            deg = self._hidden_degrees().astype(np.float64)
            probs = (deg + 1e-8) / (deg.sum() + 1e-8)
            for o in range(O):
                chosen_hidden = rng.choice(H, size=min(fan_in, H), replace=False, p=probs)
                W[I + chosen_hidden, I + H + o] = 1.0
        else:
            raise ValueError(f"Unknown output_strategy: {self.output_strategy}")

        # 4) Zero-out disallowed routes explicitly (safety)
        # No input->output or output->input or input->input or output->output
        W[0:I, I + H:T] = 0.0
        W[I + H:T, 0:I] = 0.0
        W[0:I, 0:I] = 0.0
        W[I + H:T, I + H:T] = 0.0

        return W

    def build(self, input_size: int, use_legacy_behavior: bool = None) -> ArbitraryWiring:
        """
        Return an ArbitraryWiring instance initialized with the full wiring matrix.
        
        Args:
            input_size: Input dimension (must match self.input_size)
            use_legacy_behavior: If True, use incorrect legacy behavior for testing/comparison.
                                If None, uses value set on WsFlexHiddenWiring instance (default False).
        """
        if input_size != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {input_size}")
        W = self.full_wiring_matrix()
        # Use instance attribute if not explicitly provided
        if use_legacy_behavior is None:
            use_legacy_behavior = getattr(self, '_use_legacy_behavior', False)
        return ArbitraryWiring(
            wiring_matrix=W,
            input_size=self.input_size,
            hidden_size=self._hidden_size(),
            output_size=self.output_size,
            use_legacy_behavior=use_legacy_behavior,
        )

    # -------------------------- Internals ----------------------------

    def _hidden_size(self) -> int:
        if self._is_nx():
            # Assume nodes are 0..H-1 or any set; we only need the count
            return len(self.hidden_graph.nodes())
        arr = np.asarray(self.hidden_graph)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("hidden_graph ndarray must be square HxH.")
        return int(arr.shape[0])

    def _is_nx(self) -> bool:
        if nx is None:
            return False
        return isinstance(self.hidden_graph, (nx.Graph, nx.DiGraph))

    def _hidden_block_oriented(self) -> np.ndarray:
        """Return an oriented HxH hidden adjacency block (float32)."""
        if self._is_nx():
            return self._hidden_block_from_nx()
        # ndarray path
        A = np.asarray(self.hidden_graph, dtype=self.dtype)
        H = A.shape[0]
        if self.hidden_edge_orientation == "symmetric":
            # Add both directions for any nonzero (upper/lower) entries
            A = ((A + A.T) > 0).astype(self.dtype)
        elif self.hidden_edge_orientation == "random_oriented":
            rng = np.random.default_rng(self.seed + 2)
            # Symmetrize support, then pick one direction per undirected pair
            S = ((A + A.T) > 0).astype(np.bool_)
            upper = np.triu(S, k=1)
            B = np.zeros_like(S, dtype=self.dtype)
            # For each undirected pair (i,j) with upper[i,j]=True, choose direction randomly
            us, vs = np.where(upper)
            choices = rng.integers(0, 2, size=len(us))
            for idx, (i, j) in enumerate(zip(us, vs)):
                if choices[idx] == 0:
                    B[i, j] = 1.0
                else:
                    B[j, i] = 1.0
            # Keep self-loops if present
            np.fill_diagonal(B, np.diag(S).astype(self.dtype))
            A = B
        elif self.hidden_edge_orientation == "as_is":
            # Respect the given directions/weights
            pass
        else:
            raise ValueError(f"Unknown hidden_edge_orientation: {self.hidden_edge_orientation}")

        if self.allow_signed_hidden_edges and not np.any(A < 0):
            A = self._apply_hidden_signs(A)

        return A.astype(self.dtype, copy=False)

    def _hidden_block_from_nx(self) -> np.ndarray:
        """Build HxH from a NetworkX Graph/DiGraph with optional weights."""
        G = self.hidden_graph
        # Relabel nodes to 0..H-1 if needed
        mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
        H = len(mapping)
        A = np.zeros((H, H), dtype=self.dtype)
        # Fill by edges
        if isinstance(G, nx.DiGraph):
            edge_iter = G.edges(data=True)
        else:
            edge_iter = G.edges(data=True)
        for u, v, data in edge_iter:
            w = float(data.get("weight", 1.0))
            ui, vi = mapping[u], mapping[v]
            if isinstance(G, nx.DiGraph):
                A[ui, vi] = w
            else:
                # Undirected: handle by orientation policy later
                A[ui, vi] = max(A[ui, vi], w)
                A[vi, ui] = max(A[vi, ui], w)

        # Apply orientation policy for undirected graphs
        if isinstance(G, nx.Graph):
            if self.hidden_edge_orientation == "symmetric":
                A = ((A + A.T) > 0).astype(self.dtype)
            elif self.hidden_edge_orientation == "random_oriented":
                rng = np.random.default_rng(self.seed + 3)
                S = ((A + A.T) > 0).astype(np.bool_)
                upper = np.triu(S, k=1)
                B = np.zeros_like(S, dtype=self.dtype)
                us, vs = np.where(upper)
                choices = rng.integers(0, 2, size=len(us))
                for idx, (i, j) in enumerate(zip(us, vs)):
                    if choices[idx] == 0:
                        B[i, j] = 1.0
                    else:
                        B[j, i] = 1.0
                np.fill_diagonal(B, np.diag(S).astype(self.dtype))
                A = B
            elif self.hidden_edge_orientation == "as_is":
                # For undirected "as_is", treat as symmetric
                A = ((A + A.T) > 0).astype(self.dtype)

        if self.allow_signed_hidden_edges and not np.any(A < 0):
            A = self._apply_hidden_signs(A)

        return A

    def _apply_hidden_signs(self, A: np.ndarray) -> np.ndarray:
        """Apply +/- signs to non-zero hidden edges with given inhibitory ratio."""
        if self.inhibitory_ratio <= 0.0:
            return A
        rng = np.random.default_rng(self.seed + 4)
        rows, cols = np.where(A != 0)
        m = len(rows)
        k = int(np.floor(self.inhibitory_ratio * m))
        if k > 0:
            idx = rng.choice(m, size=k, replace=False)
            A[rows[idx], cols[idx]] *= -1.0
        return A

    def _hidden_degrees(self) -> np.ndarray:
        """Return degree (undirected) or out-degree (directed) per hidden node as float array (H,)."""
        H = self._hidden_size()
        if self._is_nx():
            G = self.hidden_graph
            if isinstance(G, nx.DiGraph):
                deg = np.array([G.out_degree(n) for n in sorted(G.nodes())], dtype=np.float32)
            else:
                deg = np.array([G.degree(n) for n in sorted(G.nodes())], dtype=np.float32)
            return deg
        A = np.asarray(self.hidden_graph)
        if self.hidden_edge_orientation == "as_is":
            # Count outgoing edges
            return np.asarray((A != 0).sum(axis=1), dtype=np.float32)
        # Otherwise use symmetrized support
        S = ((A + A.T) > 0).astype(np.int32)
        return np.asarray(S.sum(axis=1), dtype=np.float32)

    # ---------------------- Convenience helpers ----------------------

    def to_dict(self) -> Dict[str, Any]:
        """Metadata useful for logging/serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self._hidden_size(),
            "output_size": self.output_size,
            "input_strategy": self.input_strategy,
            "output_strategy": self.output_strategy,
            "hidden_edge_orientation": self.hidden_edge_orientation,
            "add_hidden_self_loops": self.add_hidden_self_loops,
            "fan_in_inputs": self.fan_in_inputs,
            "fan_in_hidden_per_output": self.fan_in_hidden_per_output,
            "allow_signed_hidden_edges": self.allow_signed_hidden_edges,
            "inhibitory_ratio": self.inhibitory_ratio,
            "seed": self.seed,
        }



class ArbitraryWiring(Wiring):
    """
    Wiring class that can ingest arbitrary WS-flex graphs from architecture search.
    
    This class takes a wiring matrix and neuron configuration from the architecture
    search outputs and creates a compatible wiring structure for NCP models.
    
    IMPORTANT: In ncps, 'units' refers to internal neurons only (hidden + output),
    NOT including inputs. Inputs are external and connected via sensory_adjacency_matrix.
    """
    
    def __init__(self, 
                 wiring_matrix: np.ndarray,
                 input_size: int,
                 hidden_size: int, 
                 output_size: int,
                 neuron_types: Optional[List[str]] = None,
                 connection_weights: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 use_legacy_behavior: bool = False):
        """
        Initialize the arbitrary wiring.
        
        Args:
            wiring_matrix: Connection matrix from architecture search of shape (I+H+O, I+H+O)
            input_size: Number of input features (I)
            hidden_size: Number of hidden neurons (H)
            output_size: Number of output classes (O)
            neuron_types: List of neuron types ('sensory', 'inter', 'motor')
            connection_weights: Optional connection weights
            metadata: Additional metadata about the architecture
            logger: Optional logger for output
            use_legacy_behavior: If True, use incorrect legacy behavior (for testing/comparison)
        """
        # In ncps, 'units' = internal neurons only (hidden + output), NOT inputs
        # Legacy behavior incorrectly includes inputs in units
        if use_legacy_behavior:
            total_units = input_size + hidden_size + output_size
        else:
            total_units = hidden_size + output_size  # Internal neurons only
        
        super().__init__(units=total_units)
        
        # Set the output dimension (number of motor neurons)
        self.set_output_dim(output_size)
        
        # Store architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_units_in_matrix = input_size + hidden_size + output_size  # Full matrix size
        self.use_legacy_behavior = use_legacy_behavior
        
        # Store the wiring matrix and metadata
        self.wiring_matrix = wiring_matrix
        self.connection_weights = connection_weights if connection_weights is not None else wiring_matrix
        self.metadata = metadata or {}
        
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine neuron types if not provided
        if neuron_types is None:
            self.neuron_types = self._create_default_neuron_types()
        else:
            self.neuron_types = neuron_types
            
        # Validate the wiring matrix
        self._validate_wiring_matrix()
        
        # Build the wiring structure
        if use_legacy_behavior:
            # Legacy: incorrect behavior that includes inputs in units
            self._build_wiring_from_matrix_legacy()
        else:
            # Correct: split wiring matrix into sensory_adjacency_matrix and adjacency_matrix
            self._build_wiring_from_matrix_correct()
        
        self.logger.info(f"ArbitraryWiring initialized: {input_size}->{hidden_size}->{output_size} (legacy={use_legacy_behavior})")
    
    def _create_default_neuron_types(self) -> List[str]:
        """Create default neuron types based on layer structure."""
        neuron_types = []
        neuron_types.extend(['sensory'] * self.input_size)
        neuron_types.extend(['inter'] * self.hidden_size)
        neuron_types.extend(['motor'] * self.output_size)
        return neuron_types
    
    def _validate_wiring_matrix(self):
        """Validate the wiring matrix dimensions and structure."""
        expected_shape = (self.total_units_in_matrix, self.total_units_in_matrix)
        if self.wiring_matrix.shape != expected_shape:
            raise ValueError(f"Wiring matrix shape mismatch: expected {expected_shape}, got {self.wiring_matrix.shape}")
        
        # Check for any negative weights (which shouldn't exist in our architecture search)
        if np.any(self.wiring_matrix < 0):
            # self.logger.warning("Found negative weights in wiring matrix - converting to absolute values")
            self.wiring_matrix = np.abs(self.wiring_matrix)
    
    def _build_wiring_from_matrix_correct(self):
        """
        Build the wiring structure correctly by splitting the wiring matrix.
        
        In ncps:
        - sensory_adjacency_matrix: [input_dim, units] - connections from inputs to internal neurons
        - adjacency_matrix: [units, units] - connections between internal neurons
        
        Our wiring_matrix has shape [I+H+O, I+H+O] where:
        - I = input_size (external inputs, not part of units)
        - H = hidden_size (internal neurons)
        - O = output_size (internal neurons)
        
        We need to extract:
        - sensory_adjacency_matrix: wiring_matrix[0:I, I:I+H+O] (I->H and I->O)
        - adjacency_matrix: wiring_matrix[I:I+H+O, I:I+H+O] (H->H, H->O, but NOT I->anything)
        """
        # Matrix indices
        I = self.input_size
        H = self.hidden_size
        O = self.output_size
        I_start, I_end = 0, I
        H_start, H_end = I, I + H
        O_start, O_end = I + H, I + H + O
        
        # Initialize matrices (will be set by parent class methods)
        # But we need to ensure they're the right size
        
        # Extract sensory_adjacency_matrix: [I, H+O] = connections from inputs to internal neurons
        # This includes I->H and I->O connections
        sensory_to_internal = self.wiring_matrix[I_start:I_end, H_start:O_end]  # [I, H+O]
        
        # Extract adjacency_matrix: [H+O, H+O] = connections between internal neurons
        # This includes H->H, H->O, and potentially O->H, O->O if present in the wiring matrix
        internal_to_internal = self.wiring_matrix[H_start:O_end, H_start:O_end].copy()  # [H+O, H+O]
        
        # Note: We preserve all connections from the wiring matrix, including any O->H or O->O
        # connections that might exist. Standard NCP doesn't have these, but our wiring might.
        
        # Now populate the ncps matrices using the parent class methods
        # Note: We need to call build() first to initialize sensory_adjacency_matrix
        # But we'll populate it manually here since we have the full matrix
        
        # For sensory_adjacency_matrix, we need to add synapses using add_sensory_synapse
        # But that requires input_dim to be set, which happens in build()
        # So we'll store the matrix and populate it in build()
        self._sensory_matrix_to_populate = sensory_to_internal
        self._adjacency_matrix_to_populate = internal_to_internal
        
        # Populate adjacency_matrix using add_synapse
        # Map internal neuron indices: H neurons are 0..H-1, O neurons are H..H+O-1
        for src_internal_idx in range(H + O):
            for dest_internal_idx in range(H + O):
                weight = internal_to_internal[src_internal_idx, dest_internal_idx]
                if weight != 0:
                    polarity = 1 if weight > 0 else -1
                    self.add_synapse(src_internal_idx, dest_internal_idx, polarity)
        
        self.logger.info(f"Built correct wiring: sensory_matrix {sensory_to_internal.shape} ({np.count_nonzero(sensory_to_internal)} connections), "
                        f"adjacency_matrix {internal_to_internal.shape} ({np.count_nonzero(internal_to_internal)} connections)")
    
    def _build_wiring_from_matrix_legacy(self):
        """
        Legacy (incorrect) method that treats inputs as part of units.
        This creates dense weight matrices instead of sparse ones.
        """
        # Clear any existing synapses
        self._synapses = []
        
        # Get the indices for each layer
        input_start = 0
        input_end = self.input_size
        hidden_start = self.input_size
        hidden_end = self.input_size + self.hidden_size
        output_start = self.input_size + self.hidden_size
        output_end = self.total_units_in_matrix
        
        # Build synapses based on the wiring matrix
        for i in range(self.total_units_in_matrix):
            for j in range(self.total_units_in_matrix):
                weight = self.wiring_matrix[i, j]
                if weight > 0:  # Connection exists
                    # Determine connection type and add synapse
                    self._add_synapse_from_matrix(i, j, weight)
        
        self.logger.info(f"Built legacy wiring with {len(self._synapses)} synapses (INCORRECT - creates dense matrices)")
    
    def _add_synapse_from_matrix(self, src_idx: int, dest_idx: int, weight: float):
        """Add a synapse based on the wiring matrix."""
        # Determine the type of connection
        if src_idx < self.input_size:
            src_type = 'sensory'
        elif src_idx < self.input_size + self.hidden_size:
            src_type = 'inter'
        else:
            src_type = 'motor'
            
        if dest_idx < self.input_size:
            dest_type = 'sensory'
        elif dest_idx < self.input_size + self.hidden_size:
            dest_type = 'inter'
        else:
            dest_type = 'motor'
        
        # Add the synapse with appropriate polarity
        # For now, we'll use positive polarity for excitatory connections
        # In practice, you might want to determine this based on the weight sign or other criteria
        polarity = 1 if weight > 0 else -1
        
        # Map the matrix indices to the wiring indices
        # The wiring expects indices relative to the total units
        self.add_synapse(src_idx, dest_idx, polarity)
        
        # Log some key connections for debugging
        if (src_type == 'sensory' and dest_type == 'inter') or \
           (src_type == 'inter' and dest_type == 'motor') or \
           (src_type == 'inter' and dest_type == 'inter'):
            self.logger.debug(f"Added {src_type}->{dest_type} connection: {src_idx}->{dest_idx} (weight: {weight:.3f})")
    
    def build(self, input_shape):
        """Build the wiring with input shape - required by NCPs."""
        super().build(input_shape)
        
        if self.use_legacy_behavior:
            # Legacy behavior: incorrect mapping
            for src in range(self.input_dim):
                # Connect each input to the corresponding sensory neuron
                if src < self.input_size:
                    # Add sensory synapse from input to sensory neuron
                    self.add_sensory_synapse(src, src, 1.0)
                else:
                    # If we have more inputs than sensory neurons, distribute them
                    target_neuron = src % self.input_size
                    self.add_sensory_synapse(src, target_neuron, 1.0)
            self.logger.info(f"Connected {self.input_dim} inputs to sensory neurons (legacy)")
        else:
            # Correct behavior: populate sensory_adjacency_matrix from pre-computed matrix
            if hasattr(self, '_sensory_matrix_to_populate'):
                sensory_matrix = self._sensory_matrix_to_populate
                # sensory_matrix is [I, H+O] where I=input_size, H+O=units
                # Map each input to each internal neuron based on the matrix
                for input_idx in range(min(self.input_dim, self.input_size)):
                    for internal_neuron_idx in range(self.units):  # units = H+O
                        weight = sensory_matrix[input_idx, internal_neuron_idx]
                        if weight != 0:
                            polarity = 1 if weight > 0 else -1
                            self.add_sensory_synapse(input_idx, internal_neuron_idx, polarity)
                
                # If input_dim > input_size, map extra inputs (shouldn't happen normally)
                if self.input_dim > self.input_size:
                    for input_idx in range(self.input_size, self.input_dim):
                        # Distribute to internal neurons (use modulo to cycle)
                        target_neuron = input_idx % self.units
                        self.add_sensory_synapse(input_idx, target_neuron, 1)
                
                self.logger.info(f"Connected {self.input_dim} inputs to {self.units} internal neurons "
                               f"({np.count_nonzero(sensory_matrix)} connections)")
            else:
                # Fallback if _sensory_matrix_to_populate not set
                self.logger.warning("_sensory_matrix_to_populate not set, using fallback")
                for src in range(self.input_dim):
                    target_neuron = src % self.units if self.units > 0 else 0
                    self.add_sensory_synapse(src, target_neuron, 1.0)
    
    @property
    def num_layers(self):
        """Return the number of layers - required by WiredCfCCell."""
        if self.use_legacy_behavior:
            return 3  # Legacy: includes inputs as layer 0
        else:
            return 2  # Correct: hidden (layer 0) and output (layer 1), inputs are external
    
    def get_neurons_of_layer(self, layer_id):
        """Return neurons for each layer - required by WiredCfCCell.
        
        Returns indices relative to 'units' (internal neurons only, not including inputs).
        """
        if self.use_legacy_behavior:
            # Legacy: incorrect - includes inputs in units
            if layer_id == 0:
                return list(range(self.input_size))  # Sensory neurons (WRONG - these are external)
            elif layer_id == 1:
                return list(range(self.input_size, self.input_size + self.hidden_size))  # Inter neurons
            elif layer_id == 2:
                return list(range(self.input_size + self.hidden_size, self.total_units_in_matrix))  # Motor neurons
            else:
                raise ValueError(f"Unknown layer {layer_id}")
        else:
            # Correct: indices relative to units (H+O), inputs are external
            if layer_id == 0:
                # Layer 0: Hidden/inter neurons (indices 0..H-1 within units)
                return list(range(self.hidden_size))
            elif layer_id == 1:
                # Layer 1: Motor/output neurons (indices H..H+O-1 within units)
                return list(range(self.hidden_size, self.hidden_size + self.output_size))
            else:
                raise ValueError(f"Unknown layer {layer_id} (valid: 0-1)")
    
    def get_type_of_neuron(self, neuron_id):
        """Return the type of neuron as expected by NCPs.
        
        Args:
            neuron_id: Index relative to 'units' (internal neurons only)
        """
        if self.use_legacy_behavior:
            # Legacy: incorrect indexing
            if neuron_id < self.input_size:
                return "sensory"
            elif neuron_id < self.input_size + self.hidden_size:
                return "inter"
            else:
                return "motor"
        else:
            # Correct: neuron_id is relative to units (H+O)
            if neuron_id < self.hidden_size:
                return "inter"
            else:
                return "motor"
    
    def get_wiring_summary(self) -> Dict[str, Any]:
        """Get a summary of the wiring structure."""
        # Count connections by type
        input_start = 0
        input_end = self.input_size
        hidden_start = self.input_size
        hidden_end = self.input_size + self.hidden_size
        output_start = self.input_size + self.hidden_size
        output_end = self.total_units
        
        # Sensory layer connections
        sensory_to_sensory = np.sum(self.wiring_matrix[input_start:input_end, input_start:input_end])
        sensory_to_inter = np.sum(self.wiring_matrix[input_start:input_end, hidden_start:hidden_end])
        sensory_to_motor = np.sum(self.wiring_matrix[input_start:input_end, output_start:output_end])
        
        # Inter layer connections
        inter_to_sensory = np.sum(self.wiring_matrix[hidden_start:hidden_end, input_start:input_end])
        inter_to_inter = np.sum(self.wiring_matrix[hidden_start:hidden_end, hidden_start:hidden_end])
        inter_to_motor = np.sum(self.wiring_matrix[hidden_start:hidden_end, output_start:output_end])
        
        # Motor layer connections
        motor_to_sensory = np.sum(self.wiring_matrix[output_start:output_end, input_start:input_end])
        motor_to_inter = np.sum(self.wiring_matrix[output_start:output_end, hidden_start:hidden_end])
        motor_to_motor = np.sum(self.wiring_matrix[output_start:output_end, output_start:output_end])
        
        total_connections = np.sum(self.wiring_matrix > 0)
        total_possible = self.total_units * self.total_units
        connection_density = total_connections / total_possible
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_units': self.total_units,
            'total_connections': int(total_connections),
            'connection_density': float(connection_density),
            'connectivity_breakdown': {
                'sensory_to_sensory': float(sensory_to_sensory),
                'sensory_to_inter': float(sensory_to_inter),
                'sensory_to_motor': float(sensory_to_motor),
                'inter_to_sensory': float(inter_to_sensory),
                'inter_to_inter': float(inter_to_inter),
                'inter_to_motor': float(inter_to_motor),
                'motor_to_sensory': float(motor_to_sensory),
                'motor_to_inter': float(motor_to_inter),
                'motor_to_motor': float(motor_to_motor)
            },
            'metadata': self.metadata
        }


def load_architecture_from_file(filepath: str, logger: Optional[logging.Logger] = None, use_legacy_behavior: bool = False) -> ArbitraryWiring:
    """
    Load an architecture from a JSON file and create an ArbitraryWiring instance.
    
    Args:
        filepath: Path to the architecture JSON file
        logger: Optional logger for output
        
    Returns:
        ArbitraryWiring instance
    """
    seed = 17
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract the required fields
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']
        wiring_matrix = np.array(data['wiring_matrix'])
        
        # Optional fields
        neuron_types = data.get('neuron_types')
        connection_weights = data.get('connection_weights')
        metadata = data.get('metadata', {})
        
        # CRITICAL FIX: Determine if wiring_matrix is full [I+H+O, I+H+O] or just hidden [H, H]
        total_expected = input_size + hidden_size + output_size
        matrix_size = wiring_matrix.shape[0]
        
        if matrix_size == total_expected:
            # Full matrix: Create ArbitraryWiring directly using original sizes
            # Size mismatches will be handled by projection layers in the model
            if logger:
                logger.info(f"Detected full [I+H+O, I+H+O] matrix ({matrix_size}x{matrix_size}), "
                          f"creating ArbitraryWiring with original sizes (I={input_size}, H={hidden_size}, O={output_size})")
            
            # Create ArbitraryWiring directly with original sizes from the matrix
            # This preserves the exact wiring graph that was optimized
            return ArbitraryWiring(
                wiring_matrix=wiring_matrix,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                use_legacy_behavior=use_legacy_behavior,
                metadata=metadata
            )
            
        elif matrix_size == hidden_size:
            # Hidden-only graph: Use WsFlexHiddenWiring (original behavior)
            hidden_graph = wiring_matrix
            if logger:
                logger.info(f"Detected hidden-only [H, H] matrix ({matrix_size}x{matrix_size}), using WsFlexHiddenWiring")
            
            wiring = WsFlexHiddenWiring(
                input_size=input_size,
                hidden_graph=hidden_graph,
                output_size=output_size,
                input_strategy="degree_proportional",
                output_strategy="uniform",
                hidden_edge_orientation="random_oriented",
                add_hidden_self_loops=True,
                fan_in_inputs=None,
                fan_in_hidden_per_output=None,
                allow_signed_hidden_edges=True,
                inhibitory_ratio=0.2,
                seed=seed
            )
            wiring._use_legacy_behavior = use_legacy_behavior
            return wiring
        else:
            raise ValueError(
                f"Wiring matrix size ({matrix_size}) doesn't match expected sizes: "
                f"full matrix should be {total_expected}x{total_expected}, "
                f"hidden-only should be {hidden_size}x{hidden_size}"
            )
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading architecture from {filepath}: {e}")
        raise


def create_wiring_from_architecture_data(architecture_data: Dict[str, Any], 

                                       logger: Optional[logging.Logger] = None) -> ArbitraryWiring:
    """
    Create an ArbitraryWiring instance from architecture data dictionary.
    
    Args:
        architecture_data: Dictionary containing architecture parameters
        logger: Optional logger for output
        
    Returns:
        ArbitraryWiring instance
    """
    # Extract the required fields
    input_size = architecture_data['input_size']
    hidden_size = architecture_data['hidden_size']
    output_size = architecture_data['output_size']
    wiring_matrix = np.array(architecture_data['wiring_matrix'])
    
    # Optional fields
    neuron_types = architecture_data.get('neuron_types')
    connection_weights = architecture_data.get('connection_weights')
    metadata = architecture_data.get('metadata', {})
    
    # Create the wiring instance
    wiring = WsFlexHiddenWiring(
            input_size=input_size,
            hidden_graph=wiring_matrix,
            output_size=output_size
        )
    # wiring = ArbitraryWiring(
    #     wiring_matrix=wiring_matrix,
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     output_size=output_size,
    #     neuron_types=neuron_types,
    #     connection_weights=connection_weights,
    #     metadata=metadata,
    #     logger=logger
    # )
    
    return wiring

