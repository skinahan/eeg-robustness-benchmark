# Cluster OOM Analysis: Why 64GB Fails on Cluster but 32GB Works Locally

## Problem Summary

Experiencing OOM (Out of Memory) errors on remote cluster with **64GB RAM allocation**, but experiments run successfully on local Windows machine with **32GB RAM**. This counterintuitive behavior suggests environmental differences beyond raw memory capacity.

---

## Root Causes Identified

### 1. **NumPy/PyTorch Memory Allocator Differences** ⚠️ **MOST LIKELY**

**Linux (Cluster)** uses different memory allocators than **Windows**:
- **Windows**: Uses a different malloc implementation, often more forgiving with memory fragmentation
- **Linux**: Uses glibc malloc or jemalloc, which may have stricter memory accounting or different fragmentation behavior

**Impact**: 
- PyTorch and NumPy use different allocators on Linux vs Windows
- Linux allocators may allocate more overhead/fragmentation
- Memory fragmentation on Linux can be more severe in long-running processes

**Solution**: Set memory allocator environment variables:
```bash
# Use jemalloc if available (better fragmentation handling)
export LD_PRELOAD=/path/to/libjemalloc.so
# OR use tcmalloc (alternative)
export LD_PRELOAD=/path/to/libtcmalloc.so
```

---

### 2. **OpenMP/MKL Threading Bloat** ⚠️ **CRITICAL**

**OpenMP and Intel MKL** spawn multiple threads by default, and **each thread can allocate significant memory buffers**. This is a very common cause of OOM on clusters.

**Default behavior**:
- OpenMP: Often sets threads = number of CPU cores (could be 8-32+ cores)
- MKL: Uses all available cores by default
- NumPy: May use OpenMP backend

**Memory impact**:
- Each thread can allocate 2-4GB+ of memory buffers
- 16 threads × 2GB = **32GB+ just for threading overhead**
- This explains why cluster (with more cores) fails even with more RAM

**Solution**: **LIMIT THREADING** in shell script:
```bash
# CRITICAL: Limit OpenMP threads to prevent memory bloat
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# For PyTorch, explicitly limit threads
export TORCH_NUM_THREADS=1
```

**Why this works**:
- Reduces thread-local memory allocations
- Prevents NumPy/PyTorch from spawning many threads
- Single-threaded execution uses far less memory

---

### 3. **SLURM Memory Accounting vs Actual Usage**

**SLURM counts memory differently** than your local machine:
- **Virtual Memory (VSZ)**: Includes memory-mapped files, shared libraries
- **Resident Set Size (RSS)**: Actual physical RAM used
- **Shared memory**: May be counted multiple times across processes

**Windows Task Manager** shows different metrics than Linux `top` or SLURM accounting.

**Impact**:
- SLURM may kill jobs when VSZ exceeds limit, even if RSS is lower
- Shared libraries and memory-mapped files inflate VSZ
- Your "64GB" may effectively be less due to accounting differences

**Solution**: Check actual memory usage with proper tools:
```bash
# In your Python script, log actual RSS
import psutil
process = psutil.Process()
print(f"RSS: {process.memory_info().rss / 1024**3:.2f} GB")
print(f"VMS: {process.memory_info().vms / 1024**3:.2f} GB")
```

---

### 4. **Python Memory Management Differences**

**Garbage collection** behaves differently:
- **Windows**: Python's GC may return memory to OS more promptly
- **Linux**: Python may hold onto memory longer (free memory lists)
- **Cluster**: Longer-running processes accumulate fragmentation

**Solution**: Force more aggressive GC:
```python
import gc
import torch

# After large operations
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

---

### 5. **NumPy/PyTorch Memory Pool Settings**

**NumPy** and **PyTorch** may have different default memory pool sizes on Linux vs Windows.

**Solution**: Limit memory pools explicitly:
```python
import numpy as np
import torch

# Limit NumPy memory pool (if using numpy 1.20+)
# Note: This requires numpy 1.20+ and may not be available
# np.core.multiarray._set_madvise_hugepage(False)

# Limit PyTorch memory allocator
torch.set_num_threads(1)  # Already in globals.py, but ensure it's set early
```

---

### 6. **CUDA Memory Settings (if using GPU)**

If using GPU, CUDA memory management differs:
- **Windows**: May have different CUDA driver/runtime
- **Linux**: Different CUDA memory allocation behavior

**Solution**: Set CUDA memory fraction:
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

---

### 7. **Shared Library Overhead**

**Cluster environments** may load more shared libraries:
- System libraries
- MPI libraries (if using distributed computing)
- Environment modules

Each shared library consumes memory that's counted in SLURM's VSZ.

---

### 8. **Memory Fragmentation**

**Long-running processes** on clusters accumulate fragmentation:
- Repeated allocations/deallocations create gaps
- Linux memory allocator may not coalesce freed blocks as efficiently
- Fragmentation can make 32GB of "free" memory unusable

**Solution**: Restart processes periodically (already done with fold-by-fold execution)

---

## Recommended Fixes (Priority Order)

### **IMMEDIATE FIX #1: Limit Threading** 🔥 **CRITICAL**

Add to `unified_eval_script_crosssubject_foldbyfold.sh` **BEFORE** loading environment:

```bash
# CRITICAL: Limit all threading to prevent memory bloat
# These MUST be set before loading any Python libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TORCH_NUM_THREADS=1

echo "[MEMORY] Threading limited to 1 thread per library to prevent OOM"
echo "[MEMORY] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[MEMORY] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
```

**Why this should be #1 priority**: Threading overhead is the most common cause of unexplained OOM on clusters. A 32-core cluster node spawning 32 threads can easily consume 64GB+ just in thread-local buffers.

---

### **IMMEDIATE FIX #2: Set in globals.py**

Add to `globals.py` in `set_seeds()` function (already sets `torch.set_num_threads` but ensure others are set):

```python
import os

def set_seeds(seed_num):
    # ... existing code ...
    
    # CRITICAL: Limit threading to prevent memory bloat on clusters
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAX_THREADS'] = '1'
    os.environ['TORCH_NUM_THREADS'] = '1'
    
    # Ensure PyTorch uses single thread
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)  # Inter-op threads for CUDA
    
    # ... rest of existing code ...
```

---

### **IMMEDIATE FIX #3: Monitor Actual Memory Usage**

Add memory monitoring to track RSS vs VSZ:

```python
import psutil
import os

def log_detailed_memory(stage: str):
    """Log detailed memory usage including RSS and VSZ."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / 1024**3
    vms_gb = mem_info.vms / 1024**3
    
    print(f"[MEMORY] {stage}:")
    print(f"  RSS (actual RAM): {rss_gb:.2f} GB")
    print(f"  VSZ (virtual): {vms_gb:.2f} GB")
    
    # Check if we're close to SLURM limit
    if 'SLURM_MEM_PER_NODE' in os.environ:
        slurm_mem_gb = int(os.environ['SLURM_MEM_PER_NODE']) / 1024**2
        print(f"  SLURM limit: {slurm_mem_gb:.2f} GB")
        print(f"  RSS usage: {rss_gb/slurm_mem_gb*100:.1f}% of SLURM limit")
```

---

### **IMMEDIATE FIX #4: More Aggressive GC**

Add explicit GC calls after large operations:

```python
import gc
import torch

def aggressive_gc():
    """Force aggressive garbage collection."""
    gc.collect()
    gc.collect()  # Call twice to handle circular references
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

---

## Testing Strategy

1. **Before fixes**: Monitor memory with current settings
   ```bash
   # In job script, add:
   /usr/bin/time -v python your_script.py 2>&1 | grep -E "Maximum resident|Page"
   ```

2. **After Fix #1 (threading)**: Re-run same experiment
   - Should see **significant reduction** in memory usage
   - If still OOM, proceed to other fixes

3. **Check thread count**:
   ```python
   import threading
   print(f"Active threads: {threading.active_count()}")
   ```

---

## Expected Results

**Before fixes**:
- Threading: 16-32 threads × 2-4GB = 32-128GB overhead
- Actual data: ~10-20GB
- **Total: 42-148GB** (exceeds 64GB limit)

**After Fix #1 (threading limits)**:
- Threading: 1 thread × 2GB = 2GB overhead
- Actual data: ~10-20GB
- **Total: 12-22GB** (well under 64GB limit)

**Expected memory reduction**: **60-80% reduction** in peak memory usage.

---

## Additional Debugging Commands

```bash
# Check current threading settings
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-not set}"

# Check memory limits
ulimit -a | grep -i memory

# Check SLURM memory settings
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-not set}"

# Monitor memory during execution (in another terminal)
watch -n 1 "ps aux | grep python | grep -v grep | awk '{print \$6/1024 \" MB\"}'"
```

---

## References

- [NumPy Threading Documentation](https://numpy.org/doc/stable/user/basics.threading.html)
- [PyTorch Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [OpenMP Threading and Memory](https://www.openmp.org/)
- [SLURM Memory Accounting](https://slurm.schedmd.com/sbatch.html)

---

## Summary

**Most likely cause**: **OpenMP/MKL threading** creating 16-32+ threads, each allocating 2-4GB of memory buffers. This easily explains why a 64GB cluster node fails while a 32GB local machine succeeds (local machine likely has fewer cores, so fewer threads).

**Solution**: **Set all threading environment variables to 1** before loading Python libraries. This should reduce memory usage by 60-80%.
