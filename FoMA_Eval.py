import argparse
import logging
import os
import sys
from generate_answers import process_data
from verify_answers import verify_answers
from evaluate_results import monte_carlo_evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for Lean theorem proof (generation, verification, and evaluation)")
    
    # File paths
    parser.add_argument("--input_file", default=None,
                        help="Path to the initial input file")
    parser.add_argument("--generated_file", default=None,
                        help="Path to the output file containing generated answers")
    parser.add_argument("--verification_file", default=None,
                        help="Path to the output file containing verification results")
    parser.add_argument("--evaluation_file", default=None,
                        help="Path to the output file containing evaluation results")
    
    # Task control
    parser.add_argument("--generate", action="store_true", default=False,
                        help="Enable generation of answers")
    parser.add_argument("--verify", action="store_true", default=False,
                        help="Enable verification of generated answers")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="Enable evaluation of verification results")
    
    # Generation parameters - Add all parameters from the first script
    parser.add_argument("--model", default="/map-vepfs/keyi/model/DeepSeek-Prover-V1.5-SFT",
                        help="Path to the model used for generating answers.")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of answers to generate per process via vllm.")
    parser.add_argument("--nums_answer", type=int, default=3200,
                        help="Number of answers to generate per question.")

    # Verification parameters
    parser.add_argument("--repl_path", default="./repl",
                        help="Path to the Lean REPL used for verification")
    parser.add_argument("--lean_env_path", default="./repl/test/Mathlib",
                        help="Path to the Lean environment used for verification")
    parser.add_argument("--num_batches", default=32, type=int,
                        help="Number of parallel batches for verification")
    parser.add_argument("--session_timeout", default=600, type=int,
                        help="Timeout for interactive sessions in seconds")
    parser.add_argument("--expect_timeout", default=120, type=int,
                        help="Timeout for the expect command in seconds")
    
    # Evaluation parameters
    parser.add_argument("--n_simulations", default=50, type=int,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--n_processes", default=50, type=int,
                        help="Number of parallel processes for Monte Carlo simulation")
    parser.add_argument("--custom_sample_sizes", default=None, type=str,
                        help="Custom sampling sizes as a comma-separated list (e.g., '1,5,10,50,100')")
    
    return parser.parse_args()
def set_up_logging(level=logging.INFO):
    """Set up logging with the specified level."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    args = parse_args()
    # set_up_logging()
    # Ensure at least one task is selected
    if not (args.generate or args.verify or args.evaluate):
        print("Please select at least one task (--generate, --verify, or --evaluate)")
        return
    
    # Step 1: Generate answers
    if args.generate:
        try:
            print(f"Generating answers using model {args.model}")
            # process_data(
            #     model_path=args.model_path,
            #     input_file=args.input_file,
            #     output_file=args.generated_file,
            #     api_port=args.api_port,
            #     num_processes=args.num_processes,
            #     batch_size=args.batch_size,
            #     save_interval=args.save_interval,
            #     resume=args.resume,
            #     mode=args.mode,
            #     num_answers=args.num_answers
            # )
            process_data(
                model_path=args.model,
                input_file=args.input_file,
                output_file=args.generated_file,
                batch_size=args.n,
                num_answers=args.nums_answer
            )
            print(f"Answers have been generated and saved to {args.generated_file}")
        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return
    
    # Step 2: Verify answers
    if args.verify:
        try:
            print("Starting verification of answers")
            # Use the generated file as input if answers were generated, otherwise use the provided input file
            
            verification_input = args.generated_file 
            # Check if the input file exists
            if not os.path.exists(verification_input):
                print(f"Verification input file {verification_input} does not exist. Please check the path or generate answers first.")
                if args.evaluate:
                    print("Answer verification failed, proceeding to evaluation")
                else:
                    return
            
            verify_answers(
                input_file=verification_input,
                output_file=args.verification_file,
                repl_path=args.repl_path,
                lean_env_path=args.lean_env_path,
                num_batches=args.num_batches,
                session_timeout=args.session_timeout,
                expect_timeout=args.expect_timeout
            )
            print(f"Verification complete. Results have been saved to {args.verification_file}")
        except Exception as e:
            logging.error(f"Error during answer verification: {e}")
            return
    
    # Step 3: Evaluate verification results
    if args.evaluate:
        try:
            print("Starting evaluation of verification results")
            # Check if the verification result file exists
            if not os.path.exists(args.verification_file):
                print(f"Verification result file {args.verification_file} does not exist. Please check the path or verify answers first.")
                return
                
            # Parse custom sampling sizes
            sample_sizes = None
            if args.custom_sample_sizes:
                sample_sizes = [int(size) for size in args.custom_sample_sizes.split(',')]
            
            monte_carlo_evaluate(
                input_filepath=args.verification_file,
                output_filepath=args.evaluation_file,
                sample_sizes=sample_sizes,
                n_simulations=args.n_simulations,
                n_processes=args.n_processes
            )
            print(f"Evaluation complete. Results have been saved to {args.evaluation_file}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return
    
    print("All requested tasks have been completed successfully!")

if __name__ == "__main__":
    main()