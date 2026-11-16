import time
import os
import psutil
import tracemalloc
import threading

class MemoryTracker:
    def __init__(self, interval_seconds=0.1):
        self.interval = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.peak_vms = 0
        self.is_running = False
        self._thread = None
        self.start_time = 0
        self.end_time = 0

    def _monitor(self):
        while self.is_running:
            try:
                mem_info = self.process.memory_info()
                # RSS (Resident Set Size): Actual physical RAM used by the process
                # VMS (Virtual Memory Size): Total virtual memory allocated
                if mem_info.rss > self.peak_rss:
                    self.peak_rss = mem_info.rss
                if mem_info.vms > self.peak_vms:
                    self.peak_vms = mem_info.vms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        """Starts monitoring memory usage in a background thread."""
        print("🚀 [MemoryTracker] Started monitoring...")
        tracemalloc.start()
        self.start_time = time.perf_counter()
        print("start time: ", self.start_time)
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops monitoring and prints out the peak memory usage statistics."""
        self.is_running = False
        self.end_time = time.perf_counter()
        print("end time: ", self.end_time)
        if self._thread:
            self._thread.join()
        
        # Get internal python allocations
        current_alloc, peak_alloc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n📊 --- Memory Usage Report ---")
        print(f"🔹 Peak Physical RAM (RSS):  {self.peak_rss / (1024**2):.2f} MB")
        print(f"🔹 Peak Virtual Memory (VMS): {self.peak_vms / (1024**2):.2f} MB")
        print(f"🔹 Peak Python Objects Alloc: {peak_alloc / (1024**2):.2f} MB")
        print(f"🔹 Execution Time : {self.end_time - self.start_time:.2f} seconds")
        print("-------------------------------\n")


if __name__ == "__main__":
    # Import the function from your specific model script
    # from my_model import run_inference 
    from test_hf_eng_parakeet import run_inference

    tracker = MemoryTracker(interval_seconds=0.1)
    print("start tracking...")
    tracker.start()

    print("🤖 Running model...")
    # Call your imported function here
    run_inference() 

    tracker.stop()
    print("stopped tracking...")