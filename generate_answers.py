import re
import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import random
from tqdm import tqdm
import os
import argparse
import math
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path

def init_worker(model_path, gpu_id):
    """Initialize worker process, set GPU environment, and initialize the model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} using GPU {gpu_id}")

    global model
    model = LLM(
        model=model_path,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        seed=1,
        trust_remote_code=True,
        tensor_parallel_size=1
    )

def process_single_item(item, sampling_params, num_batches):
    """Process a single data item"""
    global model
    item['autoformalization'] = "\nComplete the following Lean 4 code:\n```lean4\n"+item['autoformalization']
    prompt = item['autoformalization']
    try:
        all_answers = []
        for _ in tqdm(range(num_batches), desc=f"Processing item {item.get('source', 'unknown')}", leave=False):
            # Generate batch answers
            model_outputs = model.generate(
                [prompt],  # Only pass one prompt
                sampling_params,
                use_tqdm=False
            )
            batch_answers = [output.text for output in model_outputs[0].outputs]
            all_answers.extend(batch_answers)
        
        # Update item
        item['answers'] = all_answers
        item['autoformalization'] = prompt
        return item
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        item['answers'] = []
        item['error'] = str(e)
        return item

def load_checkpoint(checkpoint_file):
    """Load checkpoint file"""
    try:
        with open(checkpoint_file, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_processed_items(results):
    """Get set of identifiers for processed items"""
    return {(item.get('source', ''), item.get('refined_statement', '')) for item in results}

def process_batch(args):
    """Process a batch of data"""
    start_idx, end_idx, data, sampling_params, process_id, num_batches, checkpoint_dir = args

    # Create a unique checkpoint file for each process
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_process_{process_id}.json')
    batch_results = []

    # Load this process's checkpoint
    existing_results = load_checkpoint(checkpoint_file)
    processed_items = get_processed_items(existing_results)

    for i in tqdm(range(start_idx, end_idx), desc=f"Process {os.getpid()} progress"):
        item = data[i]
        # Check if already processed
        if (item.get('source', ''), item.get('refined_statement', '')) in processed_items:
            continue
            
        result = process_single_item(item, sampling_params, num_batches)
        if result:
            batch_results.append(result)
            
            # Periodically save checkpoint
            if len(batch_results) % 10 == 0:  # Save every 10 items
                existing_results.extend(batch_results)
                with open(checkpoint_file, 'w') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                batch_results = []  # Clear saved results

    # Save remaining results
    if batch_results:
        existing_results.extend(batch_results)
        with open(checkpoint_file, 'w') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

    return checkpoint_file

def merge_checkpoints(checkpoint_files, output_file):
    """Merge results from all checkpoint files"""
    all_results = []
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            results = load_checkpoint(checkpoint_file)
            all_results.extend(results)
            # Optionally delete temporary checkpoint files
            # os.remove(checkpoint_file)

    # Save merged results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results

def process_data(
    model_path,
    input_file,
    output_file,
    api_port=8012,  # Not used but kept for compatibility
    num_processes=96,  # Not used but kept for compatibility
    batch_size=200,  # This will be used as 'n' (answers per batch)
    save_interval=16,  # Not used but kept for compatibility
    resume=True,  # Will be handled via checkpoint mechanism
    mode=None,  # Not used but kept for compatibility
    num_answers=3200  # This will be used as 'nums_answer' (total answers)
):
    """
    Process data using vLLM to generate answers.
    This function provides compatibility with the original pipeline interface.
    
    Args:
        model_path (str): Path to the model
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        api_port (int): Not used with vLLM, kept for compatibility
        num_processes (int): Not used with vLLM, kept for compatibility
        batch_size (int): Used as 'n' - number of answers per batch
        save_interval (int): Not used with vLLM, kept for compatibility
        resume (bool): Will use checkpoint mechanism
        mode (str): Not used with vLLM, kept for compatibility
        num_answers (int): Total number of answers to generate per theorem
        
    Returns:
        list: The processed data
    """
    # Setup checkpoint directory
    current_directory = os.getcwd()
    checkpoint_dir = os.path.join(current_directory, 'checkpoint_mp')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Calculate num_batches
    n = batch_size  # Use batch_size as 'n'
    nums_answer = num_answers
    num_batches = math.ceil(nums_answer / n)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        n=n,
    )
    
    # Get available GPUs
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

    if not available_gpus[0]:
        import torch
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = [int(gpu) for gpu in available_gpus]
    
    num_gpus = len(available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No available GPUs")
    
    print(f"Using {num_gpus} GPUs: {available_gpus}")
    
    # Calculate range of data for each process
    batch_size_per_gpu = len(data) // num_gpus
    if batch_size_per_gpu == 0:
        batch_size_per_gpu = 1
        num_gpus = len(data)
    
    # Prepare arguments for the process pool
    pool_args = []
    for i in range(num_gpus):
        start_idx = i * batch_size_per_gpu
        end_idx = start_idx + batch_size_per_gpu if i < num_gpus - 1 else len(data)
        pool_args.append((start_idx, end_idx, data, sampling_params, i, num_batches, checkpoint_dir))
    
    # Create process pool and assign tasks
    pools = []
    tasks = []
    
    for gpu_id in available_gpus[:num_gpus]:

        pool = Pool(
            processes=1,
            initializer=init_worker,
            initargs=(model_path, gpu_id)
        )
        pools.append(pool)
        
        task = pool.apply_async(process_batch, args=[pool_args[len(tasks)]])
        tasks.append(task)
    
    # Wait for all tasks to complete and collect checkpoint file paths
    checkpoint_files = []
    for task in tqdm(tasks, desc="Waiting for tasks to complete"):
        checkpoint_file = task.get()
        checkpoint_files.append(checkpoint_file)
    
    # Close process pools
    for pool in pools:
        pool.close()
        pool.join()
    
    # Merge results from all checkpoint files
    print("Merging results...")
    final_results = merge_checkpoints(checkpoint_files, output_file)
    
    print(f"Processing complete! Total of {len(final_results)} items processed")
    print(f"Final results saved to: {output_file}")
    
    return final_results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate answers using vLLM')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input data file')
    parser.add_argument('--generated_file', type=str, default=None,
                        help='Path to the final output file')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of answers generated per sample')
    parser.add_argument('--nums_answer', type=int, default=3200,
                        help='Total number of answers to generate per input')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    return process_data(
        model_path=args.model,
        input_file=args.input_file,
        output_file=args.generated_file,
        batch_size=args.n,
        num_answers=args.nums_answer
    )

if __name__ == "__main__":
    main()