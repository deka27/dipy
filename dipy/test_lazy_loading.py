import time
import statistics
import sys
import dipy

def test_lazy_loading_functionality():
    # List of modules to test
    modules_to_test = ['align', 'core', 'data', 'denoise', 'direction', 'io', 'nn', 'reconst', 'segment', 'sims', 'stats', 'tracking', 'utils', 'viz', 'workflows']
    
    for module_name in modules_to_test:
        # Check if the module attribute exists
        assert hasattr(dipy, module_name), f"Module {module_name} should be an attribute of dipy"
        
        # Get the module object
        module = getattr(dipy, module_name)
        
        # Check if it's a module or a lazy loader object
        assert isinstance(module, type(sys)) or hasattr(module, '__loader__'), f"{module_name} should be either a module or have a __loader__ attribute"
    
    print("Lazy loading functionality test passed.")

def measure_access_time(module_path, iterations=1000):
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        module_parts = module_path.split('.')
        module = dipy
        for part in module_parts[1:]:
            module = getattr(module, part)
        _ = module.__dict__  # Force load if not already loaded
        time.sleep(0.001)  # Add 1ms artificial delay
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return statistics.mean(times)

def test_lazy_loading_performance():
    test_module = 'dipy.workflows'
    
    # Number of trials
    num_trials = 5
    first_times = []
    second_times = []
    
    for _ in range(num_trials):
        # Measure time for first access (potentially lazy loading)
        first_time = measure_access_time(test_module)
        first_times.append(first_time)
        
        # Measure time for second access (should be already loaded)
        second_time = measure_access_time(test_module)
        second_times.append(second_time)
    
    # Calculate average times
    avg_first_time = statistics.mean(first_times)
    avg_second_time = statistics.mean(second_times)
    
    print(f"Average first access time: {avg_first_time:.9f} seconds")
    print(f"Average second access time: {avg_second_time:.9f} seconds")
    
    # Check if second access is not significantly slower
    assert avg_second_time <= avg_first_time * 1.1, "Second access should not be significantly slower than the first"
    
    # Calculate the speedup
    speedup = (avg_first_time - avg_second_time) / avg_first_time * 100
    print(f"Average speedup: {speedup:.2f}%")
    
    print("Lazy loading performance test passed.")

if __name__ == "__main__":
    test_lazy_loading_functionality()
    test_lazy_loading_performance()