# comparative_scenario_test_baseline_debug.py

import time
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime
import os
import traceback  # For detailed error logging
import random     # For seeds and choices if needed

# --- Direct Imports (no try-except blocks) ---
from data import MCCConfiguration, ExecutionTier
print("Successfully imported from data.py.")

from mcc import run_unified_test as run_two_tier_test
print("Successfully imported 2-tier test function (mcc.py).")

from mcc_extended import run_unified_test_3_tier as run_three_tier_test
print("Successfully imported 3-tier heuristic test function (mcc_extended.py).")

# Import the fixed Q-learning test function
from mcc_extended_q_learning import run_unified_test_q_learning
print("Successfully imported fixed 3-tier Q-Learning test function.")

# --- End Imports ---

def define_baseline_scenario(base_seed=456):
    """
    Defines ONLY the Baseline MCCConfiguration scenario.
    Uses the seed from the provided output for consistency.
    """
    scenarios = []
    # Scenario 1: Baseline (Balanced, 3-Tier Capable)
    # Using seed 456 as per the provided logs
    scenarios.append(MCCConfiguration(
        name="Hybrid_Test", 
        num_tasks=40, 
        num_cores=3, 
        num_edge_nodes=2, 
        num_edge_cores=2,
        bandwidth_factor=1.0, 
        power_factor=1.0, 
        rf_efficiency=1.0, 
        battery_level=80.0,
        time_constraint_multiplier=1.2, 
        seed=456
    ))
    print(f"Defined 1 scenario: Baseline (Seed: {base_seed})")
    return scenarios

def run_comparison(configurations, ql_params):
    """
    Runs the available frameworks for each configuration and collects results.
    """
    results = []
    framework_functions = {
        "2-Tier Heuristic": run_two_tier_test,
        "3-Tier Heuristic": run_three_tier_test,
        "3-Tier Q-Learning": run_unified_test_q_learning
    }

    total_configs = len(configurations)
    for i, config in enumerate(configurations):
        print(f"\n===== Running Scenario {i+1}/{total_configs}: {config.name} =====")
        # Include config details in the scenario results for reference
        scenario_results = {'config_name': config.name, 'config_details': str(config)}

        for name, func in framework_functions.items():
            print(f"\n--- Running: {name} ---")
            start_time = time.time()
            result = None  # Initialize result for this framework run
            try:
                if "Q-Learning" in name:
                    # MODIFIED: Removed unsupported parameters
                    current_ql_params = ql_params.copy()
                    result = func(config, current_ql_params)
                else:
                    # Create a copy of the config for 2-tier test if needed
                    if "2-Tier" in name:
                        config_run = deepcopy(config)
                        config_run.num_edge_nodes = 0
                        config_run.num_edge_cores = 0
                    else:
                        config_run = config  # 3-Tier heuristic uses original config

                    # Run the heuristic test
                    result = func(config_run)

                duration = time.time() - start_time
                print(f"{name} finished in {duration:.2f}s")

                # Process the result
                if result and isinstance(result, dict) and 'error' not in result:
                    # Check for essential keys before storing
                    if 'final_time' in result and 'final_energy' in result:
                        scenario_results[name] = result  # Store full result dict
                        scenario_results[f"{name}_duration"] = duration
                    else:
                        # Handle cases where the test function ran but returned incomplete data
                        error_msg = "Test function returned incomplete results (missing time/energy)."
                        print(f"  WARNING: {name} - {error_msg}")
                        scenario_results[name] = {'error': error_msg}
                        scenario_results[f"{name}_duration"] = duration
                else:
                    # Handle cases where test function returned None or an error dict
                    error_msg = result.get('error', 'Unknown error or None returned') if isinstance(result, dict) else 'Function returned None.'
                    print(f"  ERROR/SKIP: {name} - {error_msg}")
                    scenario_results[name] = {'error': error_msg}
                    scenario_results[f"{name}_duration"] = duration  # Store duration even if error occurred

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Unhandled Exception: {type(e).__name__}: {str(e)}"
                print(f"!!! {name} FAILED after {duration:.2f}s: {error_msg} !!!")
                traceback.print_exc()  # Print full traceback for debugging
                scenario_results[name] = {'error': error_msg}
                scenario_results[f"{name}_duration"] = duration

        results.append(scenario_results)
        print(f"===== Finished Scenario: {config.name} =====")

    return results


def display_results(results):
    """
    Displays the comparison results in a formatted table using tabulate.
    """
    if not results:
        print("No results to display.")
        return

    try:
        from tabulate import tabulate
        _has_tabulate = True
    except ImportError:
        print("Tabulate library not found. Install it (`pip install tabulate`) for formatted output.")
        _has_tabulate = False
        # Basic print fallback
        print("\n--- Basic Summary (Install 'tabulate' for better formatting) ---")
        for scenario_res in results:
            print(f"\n--- Scenario: {scenario_res.get('config_name', 'N/A')} ---")
            for key, value in scenario_res.items():
                if isinstance(value, dict) and ('final_time' in value or 'error' in value):
                    print(f"  Framework: {key}")
                    if 'error' not in value:
                        # Print essential metrics only for brevity in basic mode
                        print(f"    Final Time: {value.get('final_time', 'N/A'):.3f}")
                        print(f"    Final Energy: {value.get('final_energy', 'N/A'):.3f}")
                        print(f"    Deadline Met: {value.get('deadline_met', 'N/A')}")
                        print(f"    Schedule Valid: {value.get('final_schedule_valid', 'N/A')}")
                    else:
                        print(f"    ERROR: {value['error']}")
            print("-" * 30)
        return  # End here for basic print

    # --- Tabulate Output ---
    headers = ["Scenario", "Framework", "Time", "Energy", "Deadline", "Valid", "Reduc%", "L/E/C", "Optim Dur."]
    table_data = []

    framework_map = {
        "2-Tier Heuristic": "2T-H",
        "3-Tier Heuristic": "3T-H",
        "3-Tier Q-Learning": "3T-QL"
    }

    for scenario_res in results:
        scenario_name = scenario_res.get('config_name', 'Unknown')
        # Shorten scenario name for display
        scenario_name_short = scenario_name.replace("Scenario_", "").replace("Complex_", "")

        for fw_full_name, fw_short_name in framework_map.items():
            # Only include rows for frameworks that were supposed to run
            if fw_full_name not in scenario_res: continue

            res = scenario_res.get(fw_full_name)
            row = [scenario_name_short, fw_short_name]

            if res and isinstance(res, dict) and 'error' not in res:
                # Format values safely
                def fmt(val, prec=2, na_val="--"): return f"{val:.{prec}f}" if pd.notna(val) and isinstance(val, (int, float)) else na_val

                row.append(fmt(res.get('final_time')))
                row.append(fmt(res.get('final_energy')))
                row.append("MET" if res.get('deadline_met') else "MISS" if pd.notna(res.get('deadline_met')) else "--")
                row.append("OK" if res.get('final_schedule_valid') else "FAIL" if pd.notna(res.get('final_schedule_valid')) else "--")
                row.append(fmt(res.get('energy_reduction_percent'), 1))
                # Distribution String L/E/C
                l = res.get('final_local_count', '-')
                # Get edge node count from config details to decide if edge is applicable
                edge_nodes_in_config = 0
                try:  # Attempt to get edge count from stored config object first
                    cfg_obj = res.get('config')
                    if cfg_obj and isinstance(cfg_obj, MCCConfiguration):
                        edge_nodes_in_config = getattr(cfg_obj, 'num_edge_nodes', 0)
                    else:  # Fallback parsing from string
                        details_str = scenario_res.get('config_details', '')
                        import re
                        match = re.search(r'Edge: (\d+) nodes', details_str)
                        if match: edge_nodes_in_config = int(match.group(1))
                except Exception: pass  # Ignore parsing errors

                e = res.get('final_edge_count', '-') if "3-Tier" in fw_full_name and edge_nodes_in_config > 0 else "-"
                c = res.get('final_cloud_count', '-')
                row.append(f"{l}/{e}/{c}")
                # Use the duration recorded specifically for this framework run
                duration_key = f"{fw_full_name}_duration"
                row.append(fmt(scenario_res.get(duration_key, res.get('optimization_duration')), 2))  # Fallback to optim duration if top level missing

            else:
                # Append placeholders for Error/Skipped rows
                err_msg = res.get('error', 'SKIP') if isinstance(res, dict) else 'SKIP'
                # Abbreviate long error messages for the table
                if len(str(err_msg)) > 10: err_msg = str(err_msg)[:7] + "..."
                row.extend([err_msg] * (len(headers) - 2))

            table_data.append(row)
        # Add a separator row between scenarios ONLY if more than one scenario
        if len(results) > 1:
            table_data.append(['-' * len(h) for h in headers])

    # Remove the last separator row if it exists
    if len(results) > 1 and table_data: table_data.pop()

    print("\n=== Comparative Performance Summary (Baseline Scenario Debug) ===")
    # Use floatfmt in tabulate for better alignment of numbers
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f", numalign="right", stralign="left"))
    print("\nL/E/C = Final Task Distribution (Local / Edge / Cloud)")
    print("Optim Dur. = Total Duration for Optimiz./Learning Step (seconds)")

def display_paths(results):
    """ Displays the migration paths for heuristic and QL runs. """
    print("\n--- Migration Paths Debug ---")
    if not results:
        print("No results to display paths for.")
        return

    for scenario_res in results:
        print(f"\n===== Scenario: {scenario_res.get('config_name')} =====")

        # --- Print 3T Heuristic Path ---
        heuristic_res = scenario_res.get("3-Tier Heuristic")
        print("\n--- 3T Heuristic Path ---")
        if heuristic_res and isinstance(heuristic_res, dict) and 'migrations' in heuristic_res and 'error' not in heuristic_res:
            migrations = heuristic_res['migrations']
            if migrations:
                print(f"  ({len(migrations)} migration steps taken)")
                for migr in migrations:
                    # Added safety checks for potentially missing keys in migration dict
                    iter_num = migr.get('iteration', '-')
                    task_id = migr.get('task_id', '?')
                    from_idx = migr.get('from_unit_idx', '?')
                    from_tier = migr.get('from_tier', '?')
                    to_idx = migr.get('to_unit_idx', '?')
                    to_tier = migr.get('to_tier', '?')
                    t_before = migr.get('time_before', -1)
                    t_after = migr.get('time_after', -1)
                    e_before = migr.get('energy_before', -1)
                    e_after = migr.get('energy_after', -1)
                    print(f"    Iter {iter_num}: Task {task_id} from {from_idx} ({from_tier}) "
                          f"to {to_idx} ({to_tier}) | "
                          f"T: {t_before:.2f}->{t_after:.2f} | "
                          f"E: {e_before:.2f}->{e_after:.2f}")
            else:
                print("    - No migrations applied by heuristic.")
        elif heuristic_res and 'error' in heuristic_res: print(f"    ERROR: {heuristic_res['error']}")
        elif not heuristic_res: print("    SKIPPED")
        else: print("   - Migration data missing in results.")

        # --- Print 3T Q-Learning Path ---
        # MODIFIED: Updated to use 'migrations' instead of 'best_ql_path'
        ql_res = scenario_res.get("3-Tier Q-Learning")
        print("\n--- 3T Q-Learning Path ---")
        if ql_res and isinstance(ql_res, dict) and 'migrations' in ql_res and 'error' not in ql_res:
            migrations = ql_res['migrations']
            final_e = ql_res.get('final_energy', 'N/A')
            final_t = ql_res.get('final_time', 'N/A')
            print(f"  (Path to reach BestE={final_e:.2f}, BestT={final_t:.2f} - {len(migrations)} steps):")
            if migrations:
                path_limit = 40  # Show more steps if needed for debug
                for i, migr in enumerate(migrations):
                    if i < path_limit // 2 or i >= len(migrations) - path_limit // 2:
                        # MODIFIED: Updated to handle migration dictionary format correctly
                        task_id = migr.get('task_id', '?')
                        from_unit = migr.get('from_unit', '?') 
                        to_unit = migr.get('to_unit', '?')
                        time_before = migr.get('time_before', -1)
                        time_after = migr.get('time_after', -1)
                        energy_before = migr.get('energy_before', -1)
                        energy_after = migr.get('energy_after', -1)
                        # Map to_unit to tier if possible
                        tier_name = "?"
                        cfg = ql_res.get('config')
                        if cfg and isinstance(cfg, MCCConfiguration):
                            if 0 <= to_unit < cfg.num_cores: tier_name = "DEVICE" 
                            elif to_unit == cfg.num_cores: tier_name = "CLOUD"
                            else: tier_name = "EDGE"
                        print(f"    Step {i+1}: Task {task_id} from {from_unit} -> {to_unit} ({tier_name}) | "
                              f"T: {time_before:.2f}->{time_after:.2f} | "
                              f"E: {energy_before:.2f}->{energy_after:.2f}")
                    elif i == path_limit // 2 and len(migrations) > path_limit:
                        print(f"    ... ({len(migrations) - path_limit} steps omitted) ...")
            else:
                print("    - No migrations recorded (possibly failed, or initial state was best/unchanged).")
        elif ql_res and 'error' in ql_res: print(f"    ERROR: {ql_res['error']}")
        elif not ql_res: print("    SKIPPED")
        else: print("   - Migration data missing in results.")

    print("\n----------------------------------")

# --- Main Execution Block ---
if __name__ == "__main__":

    print("=== MCC Framework Comparison Script (Baseline Scenario Debug) ===")

    # 1. Define Only the Baseline Scenario
    # Use seed 456 as seen in the user's provided logs
    baseline_scenario = define_baseline_scenario(base_seed=456)

    # 2. Define Q-Learning Parameters
    # --- MODIFIED QL PARAMS ---
    # Removed unsupported parameters (reset_q_table, q_table_path)
    # Corrected Q-Learning parameters matching the existing implementation
    # Updated Q-Learning parameters with replay buffer
    ql_params = {
        'alpha': 0.1,              # Learning rate
        'gamma': 0.9,              # Discount factor
        'epsilon_start': 1.0,      # Initial exploration probability 
        'epsilon_end': 0.01,       # Final exploration probability
        'epsilon_decay': 0.9,      # Rate of exploration decay
        'time_penalty_factor': 50.0,  # Penalty factor (reduced)
        'energy_reward_factor': 15.0, # Reward factor (increased)
        'max_episodes': 500,        # Number of learning episodes
        'max_iterations': 500,      # Max iterations per episode
        'replay_buffer_size': 100, # Size of experience replay buffer
        'verbose': True,            # Print progress information
        'alpha_min': 0.01,  # Minimum learning rate
        'alpha_decay': 0.995  # Learning rate decay factor
    }

    # 3. Run Comparison on the single baseline scenario
    overall_start = time.time()
    if baseline_scenario:  # Check if scenario definition was successful
        # Pass the modified ql_params dictionary
        comparison_results = run_comparison(baseline_scenario, ql_params)
    else:
        comparison_results = []
        print("Failed to define baseline scenario. Aborting.")
    overall_duration = time.time() - overall_start

    # 4. Display Results Table
    display_results(comparison_results)

    # 5. Display Detailed Paths 
    display_paths(comparison_results)  # Call the updated function to print paths

    # 6. Explicitly print failures if any occurred
    failures_found = [r for r in comparison_results
                      for fw in ["2-Tier Heuristic", "3-Tier Heuristic", "3-Tier Q-Learning"]
                      if fw in r and isinstance(r.get(fw), dict) and 'error' in r[fw]]
    if failures_found:
        print("\n--- ERRORS Encountered During Run ---")
        for failure in failures_found:
            for fw_key, fw_res in failure.items():
                if isinstance(fw_res, dict) and 'error' in fw_res:
                    print(f"Scenario: {failure.get('config_name')}, Framework: {fw_key} -> Error: {fw_res['error']}")
        print("-------------------------------------")
    elif comparison_results:
        print("\nNo runtime errors encountered during framework execution.")

    print(f"\n=== Baseline Scenario Debug Script Finished in {overall_duration:.2f} seconds ===")