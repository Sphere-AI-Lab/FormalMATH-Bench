import json
import random
import argparse
from multiprocessing import Pool
from tqdm import tqdm

def check_correct(answers, sample_size):
    # Randomly sample a subset of answers and check if any of them has 'answer_bool' set to True
    sampled_answers = random.sample(answers, sample_size)
    return any(answer['answer_bool'] for answer in sampled_answers)

def simulate_single(args):
    data, sample_sizes = args
    all_theorems = list(data.keys())
    correct_counts = {size: 0 for size in sample_sizes}
    applicable_counts = {size: 0 for size in sample_sizes}

    for theorem in all_theorems:
        answers = data[theorem]
        num_answers = len(answers)

        for size in sample_sizes:
            # Skip sample sizes larger than the number of available answers
            if size > num_answers:
                continue
            applicable_counts[size] += 1
            if check_correct(answers, size):
                correct_counts[size] += 1

    # Calculate the success rate for each sample size
    aggregate_rates = {}
    for size in sample_sizes:
        rate = correct_counts[size] / applicable_counts[size] if applicable_counts[size] > 0 else 0
        aggregate_rates[str(size)] = rate
        print(f"size,{correct_counts[size]}")
    return aggregate_rates

def monte_carlo_evaluate(
    input_filepath, 
    output_filepath, 
    sample_sizes=None, 
    n_simulations=50, 
    n_processes=50
):
    """
    Evaluate the verification results using Monte Carlo simulation.
    
    Args:
        input_filepath (str): Path to the verification results file
        output_filepath (str): Path to save the evaluation results
        sample_sizes (list, optional): List of sample sizes to evaluate. Defaults to None.
        n_simulations (int, optional): Number of Monte Carlo simulations. Defaults to 50.
        n_processes (int, optional): Number of processes for parallel computation. Defaults to 50.
    """
    # Default sample sizes if not provided
    if sample_sizes is None:
        sample_sizes = sorted(list(range(1, 3200, 5)) + [32, 64, 128, 328, 648, 1024, 2048, 3200])
    
    # Load input data file
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aggregate_results = {}

    # Perform Monte Carlo simulation using multiprocessing
    with Pool(processes=n_processes) as pool:
        tasks = [(data, sample_sizes) for _ in range(n_simulations)]
        results = list(tqdm(pool.imap(simulate_single, tasks), total=n_simulations, desc="Monte Carlo in process"))

    # Aggregate results from each simulation run
    for sim, result in enumerate(results, start=1):
        aggregate_key = f"Aggregate_{sim}"
        aggregate_results[aggregate_key] = result

    # Save results to the output file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, ensure_ascii=False, indent=4)
    
    print(f"\nMonte Carlo simulation finished, results saved to {output_filepath}")
    return aggregate_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate theorem proof verification results")
    
    # File paths
    parser.add_argument("--input_file", default="/workspace/ky_ding/math/verify/0411/verified_stp_3200.json",
                        help="Path to verification results file")
    parser.add_argument("--output_file", default="/workspace/ky_ding/math/verify/0411/verified_stp_3200_success_rate_0414.json",
                        help="Path to evaluation results file")
    
    # Evaluation parameters
    parser.add_argument("--n_simulations", default=50, type=int,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--n_processes", default=50, type=int,
                        help="Number of processes for Monte Carlo simulation")
    parser.add_argument("--custom_sample_sizes", default=None, type=str,
                        help="Comma-separated list of custom sample sizes (e.g., '1,5,10,50,100')")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse custom sample sizes if provided
    sample_sizes = None
    if args.custom_sample_sizes:
        sample_sizes = [int(size) for size in args.custom_sample_sizes.split(',')]
    
    monte_carlo_evaluate(
        input_filepath=args.input_file,
        output_filepath=args.output_file,
        sample_sizes=sample_sizes,
        n_simulations=args.n_simulations,
        n_processes=args.n_processes
    )

if __name__ == "__main__":
    main()