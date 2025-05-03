import threading
import pexpect
import json
import os
import time
import tempfile
import re
import pdb
import heapq
import argparse
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import gc
import logging

# Interactive thread class
class InteractiveThread(threading.Thread):
    def __init__(self, session_id, repl_path, lean_env_path, initial_context=None, 
                 timeout=600, expect_timeout=120):
        super().__init__()
        self.session_id = session_id
        self.repl_path = repl_path
        self.lean_env_path = lean_env_path
        self.context = initial_context
        self.session = None
        self.expect_timeout = expect_timeout
        
        self.cmd_response_condition = threading.Event()
        self.cmd_query_condition = threading.Event()
        self.init_complete = threading.Event()
        self.response = None

        self.stop_flag = False
        self.timer = threading.Timer(timeout, self.stop) 

    def initialize_check(self):
        try:
            if self.context == None:
                initialize_check = {"cmd": "def init_check : Nat := 42"}
                self.send_cmd(initialize_check)
            self.session.expect('"env": 0}\r\n\r\n', timeout=self.expect_timeout)  # If the context contains 'sorries', it will have more keys other than 'env'
            self.init_complete.set()
        except:
            self.init_complete.set()
            print(f"Session {self.session_id}: Failed to initialize Lean REPL")
            print(self.context)
            print(self.session.before)
            self.stop()

    def send_cmd(self, cmd):
        cmd_str = json.dumps(cmd, ensure_ascii=False)
        self.session.sendline(cmd_str + '\n')

    def submit_and_receive(self, cmd):
        if self.stop_flag:
            return None

        self.init_complete.wait()
        
        self.send_cmd(cmd)
        
        self.cmd_query_condition.set()

        self.cmd_response_condition.wait()  # Wait for the response
        self.cmd_response_condition.clear()
        if self.response:
            output = self.response
            self.response = None
            return output  
        return None

    def process_responses(self):
        while not self.stop_flag:
            self.cmd_query_condition.wait()  # Wait for input 
            self.cmd_query_condition.clear()

            if self.stop_flag:  # Terminate session
                break

            try:
                self.session.expect('\r\n\r\n', timeout=self.expect_timeout)  # Filter out input; pexpect prints the input twice for unknown reasons
                self.session.expect(['\r\n\r\n', pexpect.EOF], timeout=self.expect_timeout)
                output = self.session.before.strip()
                output_dict = json.loads(output)
                self.response = output_dict
                self.cmd_response_condition.set()  

            except pexpect.TIMEOUT:
                print("Output timeout")
                self.cmd_response_condition.set()  # Prevent thread deadlock
                break  # Terminate session
            except pexpect.EOF:
                print("Session ended unexpectedly.")
                self.cmd_response_condition.set()  # Prevent thread deadlock
                break
            except json.JSONDecodeError as e:
                self.cmd_response_condition.set()  # Prevent thread deadlock
                print(output)
                break

            except Exception as e:
                print(f"Error in process_responses: {e}")
                self.cmd_response_condition.set()
                break

    def remove_last_comment(self):
        pattern = r'/--[^/]*?-/(\n*)$'
        self.context = re.sub(pattern, '', self.context, flags=re.DOTALL)

    def run(self):
        self.timer.start() 
        try:
            self.session = pexpect.spawn('bash', encoding='utf-8', cwd=self.lean_env_path)
            if self.context != None:
                self.remove_last_comment()
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
                    json.dump({"cmd": self.context}, temp, ensure_ascii=False)
                    temp.write("\n\n")
                    temp.flush()
                command = f'lake env {self.repl_path}/.lake/build/bin/repl < <(cat {temp.name} -)'
            else:
                command = f'lake env {self.repl_path}/.lake/build/bin/repl'
            
            self.session.sendline(command)
            self.initialize_check()
            self.process_responses()  # Continuously process responses
            self.stop()
    
        except Exception as e:
            print(f"Session {self.session_id}: An error occurred: {e}")
            self.init_complete.set()  
            self.stop()

    def stop(self):
        self.stop_flag = True
        self.init_complete.set()
        self.cmd_query_condition.set() 
        self.cmd_response_condition.set()
        self.timer.cancel()
        # Terminate the session
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close(force=True)
                del self.session 
            except:
                pass

# Process a proof batch
def process_batch(batch_id, item, batch_answers, context, autoformalization, 
                  repl_path, lean_env_path, session_timeout, expect_timeout):
    # Initialize interactive thread
    thread = InteractiveThread(
        batch_id,
        repl_path=repl_path,
        lean_env_path=lean_env_path,
        initial_context=context,
        timeout=session_timeout,
        expect_timeout=expect_timeout
    )
    thread.start()
    thread.init_complete.wait()  # Wait for initialization to complete

    results = []
    try:
        for answer in batch_answers:
            # Verify each answer in the batch
            verified_answer, answer_bool = process_answer(item, answer, autoformalization, thread)
            results.append({"answer": verified_answer, "answer_bool": answer_bool})  # Collect results
    finally:
        thread.stop()
        thread.join()
    
    return results

def process_answer(item, answer, autoformalization, thread):
    answer = answer.split("```")[0]
    try:
        outcome = thread.submit_and_receive({"cmd": autoformalization + answer, "env": 0})
        if outcome is None:
            return answer, False
        
        # Check for errors or incomplete content in the result
        if "messages" in outcome:
            is_error = False
            is_sorries = False
            for i in range(len(outcome["messages"])):
                if outcome["messages"][i]["severity"] == "error":
                    is_error = True
                elif outcome["messages"][i]["severity"] == "sorries" or 'sorries' in outcome.keys():
                    is_sorries = True
            if is_error or is_sorries:
                return answer, False
            else:
                return answer, True
        return answer, True
    except Exception as e:
        print(f"Error in process_answer: {e}")
        return answer, False

# Load existing progress (if available)
def load_progress_from_file(filepath):
    """
    Load a JSON progress file, attempting to recover data from incomplete JSON
    
    Args:
        filepath: The path to the JSON file
        
    Returns:
        dict: The loaded data dictionary, or an empty dictionary if loading fails
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded progress from {filepath}")
            return data
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
    return {}  

# Save data to a file
def save_to_file(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Progress saved to {filepath}")
    except Exception as e:
        print(f"Error saving to file {filepath}: {e}")

def verify_answers(
    input_file, 
    output_file, 
    repl_path="/workspace/ky_ding/math/minictx-eval/repl",
    lean_env_path="/workspace/ky_ding/math/minictx-eval/repl/test/Mathlib",
    num_batches=32,
    session_timeout=600,
    expect_timeout=120
):
    """
    Verify answers and save the results
    
    Args:
        input_file (str): Path to the input file containing answers to be verified
        output_file (str): Path to the output file to save verification results
        repl_path (str): Path to Lean REPL
        lean_env_path (str): Path to Lean environment
        num_batches (int): Number of parallel verification batches
        session_timeout (int): Timeout for interactive sessions (in seconds)
        expect_timeout (int): Timeout for expect commands (in seconds)
    
    Returns:
        dict: Verification results
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load existing data
    final_proof_dict = load_progress_from_file(output_file)

    # Load theorems to be processed
    with open(input_file, "r") as f:
        data = json.load(f)

    # Initialize thread lock
    lock = threading.Lock()

    for item in data:
        theorem_name = item["theorem_names"]

        # Skip if theorem has already been processed
        if theorem_name in final_proof_dict:
            print(f"Theorem {theorem_name} already processed. Skipping...")
            continue

        # item["autoformalization"]= "\nComplete the following Lean 4 code:\n```lean4\n"+item['autoformalization'].replace("sorry", "\n")

        autoformalization = item["autoformalization"].split("```lean4\n")[1]
        context = autoformalization.split("theorem")[0] or autoformalization.split("def")[0]
        autoformalization = autoformalization.replace(context, "", 1)
        # Allocate resources according to thread count
        answers = item["answers"]
    
        batch_size = math.ceil(len(answers) / num_batches)
        batches = [answers[i:i+batch_size] for i in range(0, len(answers), batch_size)]
        print(f"Processing {len(answers)} answers for theorem {theorem_name} in {len(batches)} batches")

        all_results = []  

        # Process batches in parallel using thread pool
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = []
            for batch_id, batch in enumerate(batches):
                futures.append(executor.submit(
                    process_batch,
                    batch_id,
                    item,
                    batch,
                    context,
                    autoformalization,
                    repl_path,
                    lean_env_path,
                    session_timeout,
                    expect_timeout
                ))

            # Collect results for each batch
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Save new theorem results to the final dictionary
        with lock:
            final_proof_dict[theorem_name] = all_results

        # Save progress
        save_to_file(output_file, final_proof_dict)

    save_to_file(output_file, final_proof_dict)
    return final_proof_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Verify Lean theorem proofs")
    
    # File paths
    parser.add_argument("--input_file", required=True,
                        help="Path to the input file containing answers to be verified")
    parser.add_argument("--output_file", required=True,
                        help="Path to the output file to save verification results")
    
    # Verification parameters
    parser.add_argument("--repl_path", default="/workspace/ky_ding/math/minictx-eval/repl",
                        help="Path to Lean REPL")
    parser.add_argument("--lean_env_path", default="/workspace/ky_ding/math/minictx-eval/repl/test/Mathlib",
                        help="Path to Lean environment")
    parser.add_argument("--num_batches", default=96, type=int,
                        help="Number of parallel verification batches")
    
    # Timeout parameters
    parser.add_argument("--session_timeout", default=600, type=int,
                      help="Timeout for interactive sessions (in seconds)")
    parser.add_argument("--expect_timeout", default=120, type=int,
                      help="Timeout for the expect command (in seconds)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        print("Starting answer verification...")
        verify_answers(
            input_file=args.input_file,
            output_file=args.output_file,
            repl_path=args.repl_path,
            lean_env_path=args.lean_env_path,
            num_batches=args.num_batches,
            session_timeout=args.session_timeout,
            expect_timeout=args.expect_timeout
        )
        print(f"Verification complete. Results have been saved to {args.output_file}")
    except Exception as e:
        logging.error(f"Error during verification: {e}")

if __name__ == "__main__":
    main()