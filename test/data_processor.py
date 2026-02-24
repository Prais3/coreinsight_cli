# data_processor.py

from bad_loop import count_unique_numbers

def process_large_dataset(dataset):
    """
    Simulates processing a large dataset.
    This function has a hidden O(N^2) list insertion bottleneck,
    and it relies on an external unoptimized function.
    """
    # RAG TEST: The AI needs context to know what this function actually does
    unique_count = count_unique_numbers(dataset)
    
    result_buffer = []
    
    # BOTTLENECK: Inserting at index 0 in a Python list forces 
    # the interpreter to shift all existing elements in memory O(N^2)
    for i in range(unique_count * 10):
        result_buffer.insert(0, i)
        
    return sum(result_buffer)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 5, 5, 6] * 10
    print("Processed sum:", process_large_dataset(sample_data))