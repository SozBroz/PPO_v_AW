"""
Memory profiling and allocation tracking utilities.
"""
import time
import tracemalloc
import numpy as np

class MemoryProfiler:
    """
    Tracks memory allocations and provides detailed reports.
    
    Features:
    - Tracks allocation points with stack traces
    - Aggregates by file/line or function
    - Monitors numpy array allocations
    """
    def __init__(self):
        self._snapshots = []
        self._start_time = time.time()
        self._numpy_allocations = []
        self._enabled = False
        
    def start(self):
        """Start tracking memory allocations."""
        tracemalloc.start(10)  # Track 10 frames
        self._enabled = True
        self._snapshots.append(tracemalloc.take_snapshot())
        
    def stop(self):
        """Stop tracking and generate final report."""
        if self._enabled:
            self._snapshots.append(tracemalloc.take_snapshot())
            tracemalloc.stop()
        self._enabled = False
        
    def track_numpy_allocation(self, shape: tuple, dtype: np.dtype):
        """Track a numpy array allocation."""
        if not self._enabled:
            return
            
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        stack = tracemalloc.get_traced_memory()[1]
        self._numpy_allocations.append({
            "time": time.time() - self._start_time,
            "size": size,
            "shape": shape,
            "dtype": str(dtype),
            "stack": stack
        })
        
    def get_report(self, group_by: str = "file"):
        """Generate memory allocation report."""
        if len(self._snapshots) < 2:
            return {"error": "Not enough snapshots for comparison"}
            
        report = {
            "total_time": time.time() - self._start_time,
            "numpy_allocations": self._numpy_allocations,
            "memory_diff": self._compare_snapshots(group_by),
            "leaks": []
        }
        
        # Detect potential leaks (objects not freed between snapshots)
        top_stats = self._snapshots[-1].compare_to(self._snapshots[0], "lineno")
        for stat in top_stats[:10]:
            if stat.size_diff > 0:
                report["leaks"].append({
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size_diff": stat.size_diff,
                    "count_diff": stat.count_diff
                })
                
        return report
        
    def _compare_snapshots(self, group_by):
        """Compare memory snapshots and group results."""
        snapshot_diff = self._snapshots[-1].compare_to(self._snapshots[0], group_by)
        return [{
            "group": stat.group,
            "size_diff": stat.size_diff,
            "count_diff": stat.count_diff
        } for stat in snapshot_diff if stat.size_diff > 0]
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()