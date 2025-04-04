#mcc_extended.py

from copy import deepcopy
import bisect
from collections import deque
import numpy as np
from heapq import heappush, heappop
import time
from dataclasses import dataclass
import random

from data import (
    ExecutionTier, SchedulingState, TaskMigrationState, core_execution_times, cloud_execution_times, edge_execution_times,
    generate_edge_task_execution_times,
    add_task_attributes, generate_configs, MCCConfiguration, generate_task_graph
)
from validation import validate_task_dependencies

@dataclass
class EdgeAssignment:
    """Tracks task assignment to an edge node"""
    edge_id: int  # Which edge node (E_m where m is 1...M)
    core_id: int  # Which core on that edge node

class Task(object):
    def __init__(self, id, pred_tasks=None, succ_task=None, core_times=None, cloud_times=None,
                 edge_times=None, # Added edge_times
                 num_cores=3, num_edge_nodes=0, num_edge_cores=0, # Added edge params
                 complexity=None, data_intensity=None): # Added characteristics
        """
        Initialize a task node in the directed acyclic graph for three-tier MCC.

        Args:
            id: Task identifier (vi in the paper)
            pred_tasks: List of immediate predecessor tasks
            succ_task: List of immediate successor tasks
            core_times: Optional dictionary mapping task IDs to core execution times
            cloud_times: Optional dictionary mapping task IDs to cloud execution times
            edge_times: Optional dictionary mapping task IDs to edge execution times {(edge_id, core_id): time}
            num_cores: Number of cores available in the device
            num_edge_nodes: Number of edge nodes
            num_edge_cores: Number of cores per edge node
            complexity: Computational complexity of the task
            data_intensity: Data intensity of the task
        """
        # Basic task graph structure
        self.id = id
        self.pred_tasks = pred_tasks or []
        self.succ_tasks = succ_task or [] # Note: succ_task arg name might be misleading if multiple successors intended

        # Task characteristics
        self.complexity = complexity if complexity is not None else random.uniform(0.5, 5.0)
        self.data_intensity = data_intensity if data_intensity is not None else random.uniform(0.2, 2.0)
        self.task_type = None  # Will be set later (compute, data, balanced)

        # ==== EXECUTION TIMES ====
        # Device execution times
        if core_times is not None:
            # Use provided dict directly
            self.local_execution_times = core_times.get(id, [])
        else:
             # Fallback to global if no dict provided
            self.local_execution_times = core_execution_times.get(id, [])

        # Cloud execution times [send, compute, receive]
        if cloud_times is not None:
            self.cloud_execution_times = cloud_times.get(id, [])
        else:
            self.cloud_execution_times = cloud_execution_times.get(id, [])

        # Edge execution times - Dictionary: {(edge_id, core_id): time}
        if edge_times is not None:
             # Use provided dict (might be pre-populated or empty)
             self.edge_execution_times = edge_times.get(id, {})
        else:
             # Fallback to global
             self.edge_execution_times = edge_execution_times.get(id, {})


        # ==== DATA TRANSFER PARAMETERS ====
        # Data sizes for all possible transfers (populated by add_task_attributes)
        self.data_sizes = {} # e.g., {'device_to_cloud': 1.2, 'device_to_edge1': 0.8, ...}

        # ==== FINISH TIMES - Device and Cloud ====
        # As defined in Section II.C (or equivalents)
        self.FT_l = 0.0    # FT_i^l:  Local core finish time
        self.FT_ws = 0.0   # FT_i^ws: Wireless sending finish time (upload complete)
        self.FT_c = 0.0    # FT_i^c:  Cloud computation finish time
        self.FT_wr = 0.0   # FT_i^wr: Wireless receiving finish time (download complete)

        # ==== FINISH TIMES - Edge Extension ====
        # Edge execution finish times (FT_i^e,m,k) - finish time on edge m, core k
        self.FT_edge = {}  # Map (edge_id, core_id) -> finish time on that edge core
        # Edge results receive time at device (FT_i^er,m) - when results from edge m arrive back at device
        self.FT_edge_receive = {} # Map edge_id -> time results received at device

        # Note: FT_edge_send (for edge->cloud or edge->edge) could be added if needed,
        # but often implicitly handled by ready times of subsequent tasks.


        # ==== READY TIMES - Device and Cloud ====
        # Ready times as defined in equations 3-6 (or equivalents)
        self.RT_l = 0.0   # RT_i^l:  Ready time for local execution
        self.RT_ws = 0.0  # RT_i^ws: Ready time for wireless sending (upload to cloud)
        self.RT_c = 0.0   # RT_i^c:  Ready time for cloud execution
        self.RT_wr = 0.0  # RT_i^wr: Ready time for receiving results (download from cloud)

        # ==== READY TIMES - Edge Extension ====
        # Ready times for edge execution (RT_i^e,m,k) - ready to start on edge m, core k
        self.RT_edge = {} # Map (edge_id, core_id) -> ready time on that edge core
        # Ready time for device-to-edge transfer for edge m (RT_i^des, m)
        self.RT_device_to_edge = {} # Map edge_id -> ready time to start sending data to edge m


        # ==== ASSIGNMENT AND EXECUTION STATE ====
        # Current execution tier
        self.execution_tier = ExecutionTier.DEVICE  # Default assumption

        # Device assignment
        self.device_core = -1  # If on device, which core (-1 = unassigned/not on device)

        # Edge assignment
        self.edge_assignment = None  # If on edge, stores EdgeAssignment(edge_id, core_id)

        # Legacy assignment value for sequence tracking compatibility
        # 0 to num_cores-1: device cores
        # num_cores: cloud
        # num_cores+1 onwards: edge units (e.g., row-major: edge1_core1, edge1_core2, ...)
        self.assignment = -2 # -2: Unassigned

        # Deprecated flag? Retained for now, prefer execution_tier
        self.is_core_task = False # True if assigned to a device core

        # ==== SCHEDULING PARAMETERS ====
        # Task scheduling state
        self.is_scheduled = SchedulingState.UNSCHEDULED
        # Task priority for scheduling
        self.priority_score = None

        # Total number of execution units (cores + cloud + edge units)
        num_edge_units = num_edge_nodes * num_edge_cores
        self.total_execution_units = num_cores + 1 + num_edge_units

        # Start times on each potential execution unit
        self.execution_unit_task_start_times = [-1.0] * self.total_execution_units

        # Final completion time for the task (when results are available where needed by successors)
        self.execution_finish_time = 0.0 # Use get_final_finish_time() for consistency

        # Kept for potential compatibility, but prefer specific FT fields
        self.completion_time = 0.0

    def get_edge_execution_time(self, edge_id, core_id):
        """Helper method to get edge execution time for a specific edge node and core."""
        return self.edge_execution_times.get((edge_id, core_id), float('inf'))

    def get_final_finish_time(self):
        """
        Get the effective finish time of this task, representing when its results
        are available for potential successors *at the device*.
        This is crucial for dependency checking.
        """
        if self.execution_tier == ExecutionTier.DEVICE:
            return self.FT_l
        elif self.execution_tier == ExecutionTier.CLOUD:
            return self.FT_wr # Results available after download
        elif self.execution_tier == ExecutionTier.EDGE:
            # Results available after download from edge
            if self.edge_assignment and self.edge_assignment.edge_id in self.FT_edge_receive:
                return self.FT_edge_receive[self.edge_assignment.edge_id]
            else:
                 # Fallback: if download time somehow not set, use edge compute finish time (less accurate)
                 if self.edge_assignment and (self.edge_assignment.edge_id, self.edge_assignment.core_id) in self.FT_edge:
                     return self.FT_edge[(self.edge_assignment.edge_id, self.edge_assignment.core_id)]
        # Default fallback if tier/times are inconsistent
        return max(self.FT_l, self.FT_wr,
                   max(self.FT_edge_receive.values()) if self.FT_edge_receive else 0.0,
                   0.0) # Ensure non-negative


    def update_assignment_and_tier(self, assignment_index, num_cores, num_edge_nodes, num_edge_cores):
        """Updates assignment index, execution tier, and specific assignments based on index."""
        self.assignment = assignment_index
        cloud_index = num_cores
        edge_start_index = num_cores + 1

        if 0 <= assignment_index < num_cores:
            self.execution_tier = ExecutionTier.DEVICE
            self.device_core = assignment_index
            self.edge_assignment = None
            self.is_core_task = True
        elif assignment_index == cloud_index:
            self.execution_tier = ExecutionTier.CLOUD
            self.device_core = -1
            self.edge_assignment = None
            self.is_core_task = False
        elif assignment_index >= edge_start_index:
            self.execution_tier = ExecutionTier.EDGE
            self.device_core = -1
            self.is_core_task = False
            # Calculate edge_id and core_id from index
            if num_edge_cores > 0:
                edge_offset = assignment_index - edge_start_index
                edge_id = (edge_offset // num_edge_cores) + 1
                core_id = (edge_offset % num_edge_cores) + 1
                if 1 <= edge_id <= num_edge_nodes and 1 <= core_id <= num_edge_cores:
                    self.edge_assignment = EdgeAssignment(edge_id=edge_id, core_id=core_id)
                else:
                    # Invalid edge index, revert? Or default? Log error.
                    print(f"Error: Invalid edge assignment index {assignment_index} calculated for Task {self.id}")
                    self.execution_tier = ExecutionTier.DEVICE # Fallback?
                    self.assignment = 0
                    self.device_core = 0
                    self.is_core_task = True
            else:
                 # Edge index used but no edge cores defined? Error.
                 print(f"Error: Edge assignment index {assignment_index} used but num_edge_cores is 0 for Task {self.id}")
                 self.execution_tier = ExecutionTier.DEVICE # Fallback
                 self.assignment = 0
                 self.device_core = 0
                 self.is_core_task = True
        else: # Should not happen (-1 or -2)
            self.execution_tier = None # Or keep previous?
            self.device_core = -1
            self.edge_assignment = None
            self.is_core_task = False

def total_time(tasks):
    """
    Implementation of total completion time calculation T_total for three-tier architecture.
    T_total = max(device_available_time(vi)) for vi in exit_tasks.

    device_available_time is when results are back at the device.
    """
    exit_tasks = [task for task in tasks if not task.succ_tasks]
    if not exit_tasks:
         # If no explicit exit tasks (e.g., single task graph), consider all tasks
         exit_tasks = tasks
    if not exit_tasks:
        return 0.0 # No tasks, no time

    max_completion_time = 0.0
    for task in exit_tasks:
        finish_time = task.get_final_finish_time() # Use helper to get time results are at device
        max_completion_time = max(max_completion_time, finish_time)

    return max_completion_time


def total_energy(tasks, device_power_profiles, rf_power, upload_rates):
    """
    Calculate total *mobile device* energy consumption across device, edge, and cloud tiers.
    Focuses on device compute energy and device RF energy for transfers.
    Edge/Cloud compute energy is NOT included as per the paper's focus.
    """
    total_mobile_energy = 0.0
    DEFAULT_SIGNAL = 70.0  # Default signal strength assumption

    for task in tasks:
        # 1) DEVICE TIER: Local execution energy on device core
        if task.execution_tier == ExecutionTier.DEVICE:
            core_id = task.device_core
            if core_id < 0 or core_id >= len(device_power_profiles):
                # print(f"Warning: Task {task.id} assigned to invalid device core {core_id}")
                continue # Skip invalid assignments

            # Get core power model
            core_info = device_power_profiles.get(core_id)
            if not core_info:
                # print(f"Warning: No power profile for device core {core_id}")
                continue

            idle_pwr = core_info.get('idle_power', 0.0)
            dyn_func = core_info.get('dynamic_power', lambda load: 0.0) # Default to 0 dynamic power if missing

            # Calculate power during task execution (assume full load)
            pwr_during_task = idle_pwr + dyn_func(1.0)

            # Get execution time on this core
            exec_time = 0.0
            if hasattr(task, 'local_execution_times') and core_id < len(task.local_execution_times):
                 exec_time = task.local_execution_times[core_id]
            elif task.FT_l > 0 and hasattr(task, 'execution_unit_task_start_times') and task.execution_unit_task_start_times[core_id] >= 0:
                 # Estimate from finish time if exec time missing
                 exec_time = task.FT_l - task.execution_unit_task_start_times[core_id]

            task_energy = pwr_during_task * exec_time
            total_mobile_energy += task_energy
            # print(f"Task {task.id} (Dev {core_id}): Time={exec_time:.2f}, Pwr={pwr_during_task:.2f}, E={task_energy:.2f}")


        # 2) EDGE TIER: Device RF energy for Device -> Edge upload
        elif task.execution_tier == ExecutionTier.EDGE:
            if not task.edge_assignment: continue # Should not happen if tier is EDGE

            edge_id = task.edge_assignment.edge_id
            data_key = f'device_to_edge{edge_id}'
            rate_key = 'device_to_edge' # Network rate key is generic for this link type

            # Get data size and rate
            data_size_mb = task.data_sizes.get(data_key, 0.0)
            # Use specific rate if available (e.g., per edge), else generic
            data_rate_mbps = upload_rates.get(f'device_to_edge{edge_id}', upload_rates.get(rate_key, 1.0)) # Default 1 Mbps

            # Calculate transfer time
            data_in_mbits = data_size_mb * 8.0
            transfer_time = data_in_mbits / data_rate_mbps if data_rate_mbps > 0 else 0.0

            # Get RF power model for device sending to edge
            rf_func = rf_power.get('device_to_edge')
            if callable(rf_func):
                radio_pwr = rf_func(data_rate_mbps, DEFAULT_SIGNAL)
            else:
                radio_pwr = 0.5 # Default power if model missing

            task_energy = radio_pwr * transfer_time
            total_mobile_energy += task_energy
            # print(f"Task {task.id} (Edge {edge_id}): UpSize={data_size_mb:.2f}, Rate={data_rate_mbps:.2f}, Time={transfer_time:.2f}, Pwr={radio_pwr:.2f}, E={task_energy:.2f}")


        # 3) CLOUD TIER: Device RF energy for Device -> Cloud upload
        elif task.execution_tier == ExecutionTier.CLOUD:
            data_key = 'device_to_cloud'
            rate_key = 'device_to_cloud'

            # Get data size and rate
            data_size_mb = task.data_sizes.get(data_key, 0.0)
            data_rate_mbps = upload_rates.get(rate_key, 1.0) # Default 1 Mbps

            # Calculate transfer time (should ideally match cloud_execution_times[0])
            data_in_mbits = data_size_mb * 8.0
            transfer_time = data_in_mbits / data_rate_mbps if data_rate_mbps > 0 else 0.0
            # Sanity check/override with scheduled time if available
            if hasattr(task, 'cloud_execution_times') and task.cloud_execution_times and task.cloud_execution_times[0] > 0:
                scheduled_send_time = task.cloud_execution_times[0]
                # If calculated time differs significantly, log warning but use scheduled time for consistency
                # if abs(transfer_time - scheduled_send_time) > 0.1:
                #     print(f"Warning Task {task.id}: Calculated send time {transfer_time:.2f} != scheduled {scheduled_send_time:.2f}")
                transfer_time = scheduled_send_time


            # Get RF power model for device sending to cloud
            rf_func = rf_power.get('device_to_cloud')
            if callable(rf_func):
                radio_pwr = rf_func(data_rate_mbps, DEFAULT_SIGNAL)
            else:
                radio_pwr = 0.7 # Default power if model missing

            task_energy = radio_pwr * transfer_time
            total_mobile_energy += task_energy
            # print(f"Task {task.id} (Cloud): UpSize={data_size_mb:.2f}, Rate={data_rate_mbps:.2f}, Time={transfer_time:.2f}, Pwr={radio_pwr:.2f}, E={task_energy:.2f}")

    return total_mobile_energy

def primary_assignment(tasks, num_cores=3, num_edge_nodes=0, num_edge_cores=0, upload_rates=None, download_rates=None):
    """
    Assign each task preliminarily to the tier (device, edge, or cloud)
    that yields the minimal estimated single-task completion time, ignoring concurrency.
    This provides an initial guess for the scheduling algorithm.
    """
    # Use default rates if none provided
    upload_rates = upload_rates or {}
    download_rates = download_rates or {}

    # Process each task independently
    for task in tasks:
        best_finish_time = float('inf')
        best_tier = None
        best_details = {} # To store core_id or edge_assignment

        # 1. Evaluate Device Execution
        min_local_time = float('inf')
        best_local_core = -1
        if hasattr(task, 'local_execution_times') and task.local_execution_times:
            for core_id, exec_time in enumerate(task.local_execution_times):
                 if core_id < num_cores and exec_time < min_local_time:
                     min_local_time = exec_time
                     best_local_core = core_id

            if min_local_time < best_finish_time:
                 best_finish_time = min_local_time
                 best_tier = ExecutionTier.DEVICE
                 best_details = {'core_id': best_local_core}

        # 2. Evaluate Edge Execution (for each edge node/core)
        if num_edge_nodes > 0 and num_edge_cores > 0 and hasattr(task, 'edge_execution_times'):
            for edge_id in range(1, num_edge_nodes + 1):
                 for core_id in range(1, num_edge_cores + 1):
                    edge_core_key = (edge_id, core_id)
                    exec_time = task.get_edge_execution_time(edge_id, core_id)
                    if exec_time == float('inf'): continue

                    # Estimate transfer overheads
                    # Device -> Edge Upload
                    up_data_key = f'device_to_edge{edge_id}'
                    up_rate_key = 'device_to_edge'
                    up_size_mb = task.data_sizes.get(up_data_key, 0.0)
                    up_rate_mbps = upload_rates.get(f'device_to_edge{edge_id}', upload_rates.get(up_rate_key, 1.0))
                    upload_time = (up_size_mb * 8.0) / up_rate_mbps if up_rate_mbps > 0 else 0.0

                    # Edge -> Device Download
                    down_data_key = f'edge{edge_id}_to_device'
                    down_rate_key = 'edge_to_device'
                    down_size_mb = task.data_sizes.get(down_data_key, 0.0)
                    down_rate_mbps = download_rates.get(f'edge{edge_id}_to_device', download_rates.get(down_rate_key, 1.0))
                    download_time = (down_size_mb * 8.0) / down_rate_mbps if down_rate_mbps > 0 else 0.0

                    total_edge_time = upload_time + exec_time + download_time

                    if total_edge_time < best_finish_time:
                         best_finish_time = total_edge_time
                         best_tier = ExecutionTier.EDGE
                         best_details = {'edge_id': edge_id, 'core_id': core_id}

        # 3. Evaluate Cloud Execution
        if hasattr(task, 'cloud_execution_times') and len(task.cloud_execution_times) >= 3:
             # Use pre-calculated/estimated phases including transfer times based on rates/sizes
             # T_re = T_s + T_c + T_r (already incorporates network effects)
             t_send, t_compute, t_receive = task.cloud_execution_times[:3]
             total_cloud_time = t_send + t_compute + t_receive

             if total_cloud_time < best_finish_time:
                 best_finish_time = total_cloud_time
                 best_tier = ExecutionTier.CLOUD
                 best_details = {} # No specific core/node needed for cloud


        # Apply the best assignment found
        if best_tier is not None:
            task.execution_tier = best_tier
            if best_tier == ExecutionTier.DEVICE:
                task.is_core_task = True
                task.device_core = best_details.get('core_id', -1)
                task.edge_assignment = None
            elif best_tier == ExecutionTier.EDGE:
                task.is_core_task = False
                task.device_core = -1
                task.edge_assignment = EdgeAssignment(edge_id=best_details['edge_id'], core_id=best_details['core_id'])
            else: # CLOUD
                task.is_core_task = False
                task.device_core = -1
                task.edge_assignment = None
        else:
             # Fallback if no option was viable (should not happen with defaults)
             task.execution_tier = ExecutionTier.DEVICE
             task.is_core_task = True
             task.device_core = 0 # Assign to first core
             task.edge_assignment = None
             print(f"Warning: No primary assignment found for Task {task.id}, defaulting to Device Core 0.")


        # Initialize FT times based on primary assignment (rough estimate, scheduler refines this)
        if task.execution_tier == ExecutionTier.DEVICE:
            task.FT_l = best_finish_time
            task.FT_ws = task.FT_c = task.FT_wr = 0.0
            task.FT_edge.clear()
            task.FT_edge_receive.clear()
        elif task.execution_tier == ExecutionTier.EDGE:
            task.FT_l = 0.0
            task.FT_ws = task.FT_c = task.FT_wr = 0.0
            # Store estimated finish time for the assigned edge
            # Note: Scheduler will calculate definitive times
            if task.edge_assignment:
                 e_id = task.edge_assignment.edge_id
                 c_id = task.edge_assignment.core_id
                 # Estimate edge execution finish and device receive finish
                 # These are placeholders; the scheduler computes actual FTs
                 # task.FT_edge[(e_id, c_id)] = upload_time + exec_time # Rough estimate
                 task.FT_edge_receive[e_id] = best_finish_time # Rough estimate
            else: # Clear if assignment failed
                 task.FT_edge.clear()
                 task.FT_edge_receive.clear()
        else: # CLOUD
            task.FT_l = 0.0
            # Set cloud phase times based on estimates
            if len(task.cloud_execution_times) >= 3:
                 t_s, t_c, t_r = task.cloud_execution_times[:3]
                 task.FT_ws = t_s
                 task.FT_c = t_s + t_c
                 task.FT_wr = t_s + t_c + t_r
            else: # Fallback
                 task.FT_ws = task.FT_c = task.FT_wr = best_finish_time
            task.FT_edge.clear()
            task.FT_edge_receive.clear()


def task_prioritizing(tasks):
    """
    Implements the "Task Prioritizing" phase for three-tier architecture.
    Calculates priority levels for each task based on the critical path method,
    considering the estimated execution cost in its primary assigned tier.

    Args:
        tasks: List of Task objects with primary assignments (execution_tier set)

    Modifies:
        Sets priority_score attribute on each task object

    Returns:
        Dictionary mapping task IDs to their computed priority scores
    """
    num_tasks = len(tasks)
    w = [0.0] * num_tasks # Computation costs

    # Step 1: Calculate computation costs (wi) for each task based on primary tier
    for i, task in enumerate(tasks):
        if task.execution_tier == ExecutionTier.CLOUD:
            # Cloud tasks: Use total remote time (send + compute + receive)
            if hasattr(task, 'cloud_execution_times') and len(task.cloud_execution_times) >= 3:
                 w[i] = sum(task.cloud_execution_times[:3])
            else: # Fallback
                 w[i] = task.FT_wr if hasattr(task, 'FT_wr') else 1.0

        elif task.execution_tier == ExecutionTier.EDGE:
            # Edge tasks: Use estimated total edge time (upload + exec + download)
            # This uses the times estimated during primary_assignment
            if task.edge_assignment:
                 edge_id = task.edge_assignment.edge_id
                 # Use FT_edge_receive as it represents time until result is back at device
                 w[i] = task.FT_edge_receive.get(edge_id, float('inf'))
                 # Fallback if FT_edge_receive wasn't set properly
                 if w[i] == float('inf'):
                     core_id = task.edge_assignment.core_id
                     exec_time = task.get_edge_execution_time(edge_id, core_id)
                     # Re-estimate roughly if needed
                     # This requires network info, which isn't available here. Use rough exec time.
                     w[i] = exec_time if exec_time != float('inf') else 1.0
            else: # Fallback
                w[i] = 1.0

        else: # ExecutionTier.DEVICE
            # Local tasks: Use average execution time across *available* cores
            if hasattr(task, 'local_execution_times') and task.local_execution_times:
                 num_cores_for_task = len(task.local_execution_times)
                 w[i] = sum(task.local_execution_times) / num_cores_for_task if num_cores_for_task > 0 else 1.0
            else: # Fallback
                 w[i] = task.FT_l if hasattr(task, 'FT_l') and task.FT_l > 0 else 1.0


    # Cache for memoization of priority calculations
    computed_priority_scores = {}

    def calculate_priority(task_obj):
        """Recursive priority calculation (Eq 15 & 16)."""
        task_id = task_obj.id
        if task_id in computed_priority_scores:
            return computed_priority_scores[task_id]

        task_index = task_id - 1 # 0-based index for w array

        # Base case: Exit tasks (Eq 16)
        if not task_obj.succ_tasks:
            priority = w[task_index]
            computed_priority_scores[task_id] = priority
            return priority

        # Recursive case: Non-exit tasks (Eq 15)
        max_succ_priority = 0.0
        if task_obj.succ_tasks:
             max_succ_priority = max(calculate_priority(successor) for successor in task_obj.succ_tasks)

        priority = w[task_index] + max_succ_priority
        computed_priority_scores[task_id] = priority
        return priority

    # Calculate priorities for all tasks
    for task in tasks:
        if task.id not in computed_priority_scores:
            calculate_priority(task)

    # Update priority scores in task objects
    for task in tasks:
        task.priority_score = computed_priority_scores.get(task.id, 0.0) # Assign calculated priority

    return computed_priority_scores

class InitialTaskScheduler:
    """
    Implements the initial scheduling algorithm for three-tier architecture.
    Extends the original two-tier scheduler to include edge computing capabilities.
    Focuses on minimizing completion time (makespan).
    """

    def __init__(self, tasks, num_cores=3, num_edge_nodes=0, num_edge_cores=0, # Added edge params
                 upload_rates=None, download_rates=None): # Added network params
        """
        Initialize scheduler with tasks and resources across all three tiers.

        Args:
            tasks: List of Task objects to schedule
            num_cores: Number of cores in the mobile device (K)
            num_edge_nodes: Number of edge computing nodes available (M)
            num_edge_cores: Number of cores per edge node
            upload_rates: Dictionary of upload bandwidths (Mbps)
            download_rates: Dictionary of download bandwidths (Mbps)
        """
        self.tasks = tasks
        self.num_cores = num_cores  # Device cores (K)
        self.num_edge_nodes = num_edge_nodes # Edge nodes (M)
        self.num_edge_cores = num_edge_cores # Cores per edge node

        # Network rates (essential for edge scheduling)
        self.upload_rates = upload_rates or {}
        self.download_rates = download_rates or {}

        # Calculate total execution units for sequence tracking
        self.cloud_index = self.num_cores
        self.edge_start_index = self.cloud_index + 1
        self.num_edge_units = self.num_edge_nodes * self.num_edge_cores
        self.total_units = self.edge_start_index + self.num_edge_units

        # --- RESOURCE AVAILABILITY TRACKING ---
        # Device cores
        self.core_earliest_ready = [0.0] * self.num_cores # EST for each device core k

        # Cloud resources (simplified view from device perspective)
        self.cloud_upload_channel_ready = 0.0 # EST for device->cloud channel
        self.cloud_download_channel_ready = 0.0 # EST for cloud->device channel
        # Note: Cloud compute availability is assumed infinite/handled by cloud

        # Edge compute resources
        # EST for each edge core (edge_id, core_id)
        self.edge_cores_earliest_ready = {} # Map (edge_id, core_id) -> availability time

        # Edge communication channels (view from device)
        self.device_to_edge_channels_ready = {}  # Map edge_id -> EST for device->edge_id channel
        self.edge_to_device_channels_ready = {}  # Map edge_id -> EST for edge_id->device channel
        # Note: Edge<->Edge and Edge<->Cloud channels aren't directly tracked here,
        # as scheduling decisions are primarily device-centric. Their effects are
        # captured in the ready times of dependent tasks.

        # Initialize edge resource tracking
        for e_id in range(1, self.num_edge_nodes + 1):
            self.device_to_edge_channels_ready[e_id] = 0.0
            self.edge_to_device_channels_ready[e_id] = 0.0
            for c_id in range(1, self.num_edge_cores + 1):
                self.edge_cores_earliest_ready[(e_id, c_id)] = 0.0

        # --- SEQUENCE TRACKING ---
        # Initialize execution sequences (Sk) for all resources
        # Indices: 0..K-1 (device cores), K (cloud), K+1.. (edge units)
        self.sequences = [[] for _ in range(self.total_units)]

    def get_edge_unit_index(self, edge_id, core_id):
        """Maps an edge (node_id, core_id) to its index in the sequences list."""
        if not (1 <= edge_id <= self.num_edge_nodes and 1 <= core_id <= self.num_edge_cores):
            return -1 # Invalid ID
        # Calculate 0-based offset from the start of edge indices
        offset = (edge_id - 1) * self.num_edge_cores + (core_id - 1)
        return self.edge_start_index + offset

    def get_priority_ordered_tasks(self):
        """Orders tasks by priority score (descending)."""
        # Create list of (priority, id) tuples, handling None scores
        task_priority_list = [(task.priority_score if task.priority_score is not None else 0, task.id)
                              for task in self.tasks]
        task_priority_list.sort(key=lambda x: x[0], reverse=True) # Sort by priority descending
        return [item[1] for item in task_priority_list] # Return just the ordered task IDs

    def classify_entry_tasks(self, priority_order):
        """Separates tasks into entry and non-entry based on predecessors."""
        entry_tasks = []
        non_entry_tasks = []
        task_map = {task.id: task for task in self.tasks} # Quick lookup

        for task_id in priority_order:
            task = task_map.get(task_id)
            if not task: continue

            if not task.pred_tasks:
                entry_tasks.append(task)
            else:
                non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def _get_predecessor_finish_time_at_device(self, pred_task):
        """
        Helper: Determines when a predecessor task's results are available *at the device*.
        This is crucial for calculating ready times of successor tasks starting on device,
        or tasks needing data *from* the device for upload (to edge or cloud).
        """
        if not pred_task: return 0.0

        # Use the dedicated helper method in Task class
        return pred_task.get_final_finish_time()


    def calculate_ready_times(self, task):
        """
        Calculates ready times for a task across all potential starting points:
        - RT_l: Ready to start on a device core.
        - RT_ws: Ready to start uploading to cloud.
        - RT_device_to_edge[m]: Ready to start uploading to edge node m.
        - RT_edge[m,k]: Ready to start computing on edge node m, core k.
        """
        # Find latest time results from ANY predecessor are available AT THE DEVICE
        max_pred_finish_at_device = 0.0
        if task.pred_tasks:
            max_pred_finish_at_device = max(
                self._get_predecessor_finish_time_at_device(pred) for pred in task.pred_tasks
            )

        # 1. Device Ready Time (RT_l)
        # Can start on device core only after all predecessors' results are available at device.
        task.RT_l = max_pred_finish_at_device

        # 2. Cloud Upload Ready Time (RT_ws)
        # Can start uploading to cloud after predecessors' results are available at device
        # AND the device->cloud upload channel is free.
        task.RT_ws = max(max_pred_finish_at_device, self.cloud_upload_channel_ready)

        # 3. Device-to-Edge Upload Ready Times (RT_device_to_edge)
        # Can start uploading to edge 'e_id' after predecessors' results available at device
        # AND the device->edge_id upload channel is free.
        if not hasattr(task, 'RT_device_to_edge'): task.RT_device_to_edge = {}
        for e_id in range(1, self.num_edge_nodes + 1):
            task.RT_device_to_edge[e_id] = max(max_pred_finish_at_device,
                                              self.device_to_edge_channels_ready.get(e_id, 0.0))

        # 4. Edge Execution Ready Times (RT_edge) - This depends on upload finishing.
        # This is calculated *during* edge evaluation, not here. Initialize dict.
        if not hasattr(task, 'RT_edge'): task.RT_edge = {}


    def identify_optimal_local_core(self, task):
        """
        Finds optimal device core assignment for a task to minimize finish time.
        Considers task's RT_l and core availability.
        """
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')
        ready_time = task.RT_l # Ready time for local execution

        for core_id in range(self.num_cores):
            # Check if task has execution time defined for this core
            if core_id >= len(task.local_execution_times): continue

            exec_time = task.local_execution_times[core_id]
            if exec_time == float('inf'): continue

            # Calculate earliest possible start time on this core
            start_time = max(ready_time, self.core_earliest_ready[core_id])
            finish_time = start_time + exec_time

            # Track the core with earliest finish time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core_id
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time


    def evaluate_cloud_option(self, task):
        """
        Evaluates the finish time if the task executes on the cloud.
        Considers task's RT_ws and channel/cloud availability.
        """
        if not task.cloud_execution_times or len(task.cloud_execution_times) < 3:
            return float('inf'), {} # Invalid cloud data

        t_send, t_compute, t_receive = task.cloud_execution_times[:3]
        ready_time = task.RT_ws # Ready time for cloud upload

        # Phase 1: Upload (Device -> Cloud)
        upload_start = max(ready_time, self.cloud_upload_channel_ready)
        upload_finish = upload_start + t_send

        # Phase 2: Cloud Compute
        # Ready time depends on upload finishing. Cloud resource assumed available.
        compute_start = upload_finish
        compute_finish = compute_start + t_compute

        # Phase 3: Download (Cloud -> Device)
        # Ready time depends on compute finishing.
        download_ready = compute_finish
        download_start = max(download_ready, self.cloud_download_channel_ready)
        download_finish = download_start + t_receive # This is FT_wr

        timing_details = {
            'upload_start': upload_start, 'upload_finish': upload_finish,
            'compute_start': compute_start, 'compute_finish': compute_finish,
            'download_start': download_start, 'download_finish': download_finish
        }

        return download_finish, timing_details

    def evaluate_edge_option(self, task, edge_id, core_id):
        """
        Evaluates the finish time if the task executes on a specific edge core.
        Considers device->edge upload, edge exec, edge->device download times,
        and relevant resource availabilities.
        """
        edge_core_key = (edge_id, core_id)
        exec_time = task.get_edge_execution_time(edge_id, core_id)
        if exec_time == float('inf'):
             return float('inf'), {}

        # Ready time for upload to this edge
        upload_ready_time = task.RT_device_to_edge.get(edge_id, 0.0)

        # Phase 1: Upload (Device -> Edge)
        up_data_key = f'device_to_edge{edge_id}'
        up_rate_key = 'device_to_edge'
        up_size_mb = task.data_sizes.get(up_data_key, 0.0)
        up_rate_mbps = self.upload_rates.get(f'device_to_edge{edge_id}', self.upload_rates.get(up_rate_key, 1.0))
        upload_time = (up_size_mb * 8.0) / up_rate_mbps if up_rate_mbps > 0 else 0.0

        upload_start = max(upload_ready_time, self.device_to_edge_channels_ready.get(edge_id, 0.0))
        upload_finish = upload_start + upload_time

        # Phase 2: Edge Execution
        # Ready time depends on upload finishing.
        exec_ready_time = upload_finish
        task.RT_edge[edge_core_key] = exec_ready_time

        exec_start = max(exec_ready_time, self.edge_cores_earliest_ready.get(edge_core_key, 0.0))
        exec_finish = exec_start + exec_time # This is FT_edge for this core

        # Phase 3: Download (Edge -> Device)
        down_data_key = f'edge{edge_id}_to_device'
        down_rate_key = 'edge_to_device'
        down_size_mb = task.data_sizes.get(down_data_key, 0.0)
        down_rate_mbps = self.download_rates.get(f'edge{edge_id}_to_device', self.download_rates.get(down_rate_key, 1.0))
        download_time = (down_size_mb * 8.0) / down_rate_mbps if down_rate_mbps > 0 else 0.0

        download_ready = exec_finish
        download_start = max(download_ready, self.edge_to_device_channels_ready.get(edge_id, 0.0))
        download_finish = download_start + download_time # This is FT_edge_receive for this edge

        timing_details = {
            'upload_start': upload_start, 'upload_finish': upload_finish,
            'exec_start': exec_start, 'exec_finish': exec_finish,
            'download_start': download_start, 'download_finish': download_finish,
            'upload_time': upload_time, 'exec_time': exec_time, 'download_time': download_time
        }

        return download_finish, timing_details


    def identify_best_execution_option(self, task):
        """
        Compares execution options across device, edge, and cloud, selecting the one
        that minimizes the task's final completion time (results back at device).
        """
        best_option = {'tier': None, 'finish_time': float('inf'), 'details': {}}

        # 1. Evaluate Device Cores
        core_id, start_time, finish_time = self.identify_optimal_local_core(task)
        if core_id != -1 and finish_time < best_option['finish_time']:
            best_option = {
                'tier': ExecutionTier.DEVICE,
                'finish_time': finish_time,
                'details': {'core_id': core_id, 'start_time': start_time, 'finish_time': finish_time}
            }

        # 2. Evaluate Edge Cores
        if self.num_edge_nodes > 0 and self.num_edge_cores > 0:
             for e_id in range(1, self.num_edge_nodes + 1):
                 for c_id in range(1, self.num_edge_cores + 1):
                    finish_time, details = self.evaluate_edge_option(task, e_id, c_id)
                    if finish_time < best_option['finish_time']:
                         best_option = {
                             'tier': ExecutionTier.EDGE,
                             'finish_time': finish_time,
                             'details': {'edge_id': e_id, 'core_id': c_id, **details}
                         }

        # 3. Evaluate Cloud
        finish_time, details = self.evaluate_cloud_option(task)
        if finish_time < best_option['finish_time']:
            best_option = {
                'tier': ExecutionTier.CLOUD,
                'finish_time': finish_time,
                'details': details
            }

        return best_option


    def schedule_task(self, task, chosen_option):
        """Schedules the task based on the chosen best option."""
        tier = chosen_option['tier']
        details = chosen_option['details']
        task_id = task.id
        sequence_index = -1 # Index in self.sequences

        if tier == ExecutionTier.DEVICE:
            core_id = details['core_id']
            start_time = details['start_time']
            finish_time = details['finish_time']
            sequence_index = core_id

            # Update task state
            task.update_assignment_and_tier(core_id, self.num_cores, self.num_edge_nodes, self.num_edge_cores)
            task.FT_l = finish_time
            task.execution_finish_time = finish_time # Final time is local finish time
            task.execution_unit_task_start_times[core_id] = start_time
            task.FT_ws = task.FT_c = task.FT_wr = 0.0 # Clear other tiers
            task.FT_edge.clear()
            task.FT_edge_receive.clear()


            # Update resource availability
            self.core_earliest_ready[core_id] = finish_time

        elif tier == ExecutionTier.EDGE:
            edge_id = details['edge_id']
            core_id = details['core_id']
            edge_core_key = (edge_id, core_id)
            sequence_index = self.get_edge_unit_index(edge_id, core_id)

            # Update task state
            task.update_assignment_and_tier(sequence_index, self.num_cores, self.num_edge_nodes, self.num_edge_cores)
            task.FT_edge[edge_core_key] = details['exec_finish']
            task.FT_edge_receive[edge_id] = details['download_finish']
            task.execution_finish_time = details['download_finish'] # Final time is when results arrive at device
            task.execution_unit_task_start_times[sequence_index] = details['exec_start']
            task.FT_l = task.FT_ws = task.FT_c = task.FT_wr = 0.0 # Clear other tiers


            # Update resource availability
            self.device_to_edge_channels_ready[edge_id] = details['upload_finish']
            self.edge_cores_earliest_ready[edge_core_key] = details['exec_finish']
            self.edge_to_device_channels_ready[edge_id] = details['download_finish']

        elif tier == ExecutionTier.CLOUD:
            sequence_index = self.cloud_index

            # Update task state
            task.update_assignment_and_tier(self.cloud_index, self.num_cores, self.num_edge_nodes, self.num_edge_cores)
            task.FT_ws = details['upload_finish']
            task.FT_c = details['compute_finish']
            task.FT_wr = details['download_finish']
            task.execution_finish_time = details['download_finish'] # Final time is when results arrive at device
            task.execution_unit_task_start_times[self.cloud_index] = details['upload_start']
            task.FT_l = 0.0 # Clear other tiers
            task.FT_edge.clear()
            task.FT_edge_receive.clear()

            # Update resource availability
            self.cloud_upload_channel_ready = details['upload_finish']
            # Cloud compute assumed infinite, no update needed for availability
            self.cloud_download_channel_ready = details['download_finish']

        else:
            print(f"Error: Cannot schedule task {task_id}, no valid tier chosen.")
            return # Skip scheduling this task


        # Mark task as initially scheduled and add to sequence
        task.is_scheduled = SchedulingState.SCHEDULED
        if 0 <= sequence_index < len(self.sequences):
             self.sequences[sequence_index].append(task_id)
        else:
             print(f"Error: Invalid sequence index {sequence_index} for task {task_id}")


    def run_initial_scheduling(self):
        """Executes the entire initial scheduling process."""
        # Order tasks by priority
        priority_ordered_ids = self.get_priority_ordered_tasks()
        task_map = {task.id: task for task in self.tasks}

        # Process tasks in priority order
        for task_id in priority_ordered_ids:
            task = task_map.get(task_id)
            if not task: continue

            # Calculate ready times based on already scheduled predecessors
            self.calculate_ready_times(task)

            # Identify the best execution option (Device, Edge, or Cloud)
            best_option = self.identify_best_execution_option(task)

            # Schedule the task based on the chosen option
            self.schedule_task(task, best_option)

        return self.sequences


# Standalone function to wrap the initial scheduling process
def execution_unit_selection(tasks, num_cores=3, num_edge_nodes=0, num_edge_cores=0, upload_rates=None, download_rates=None):
    """
    Performs the complete initial scheduling (minimal-delay) for three-tier architecture.
    Includes primary assignment, prioritizing, and execution unit selection.

    Args:
        tasks: List of Task objects
        num_cores: Number of device cores
        num_edge_nodes: Number of edge nodes
        num_edge_cores: Number of cores per edge node
        upload_rates: Upload bandwidths
        download_rates: Download bandwidths

    Returns:
        sequences: List of task sequences Sk for each execution unit (device, cloud, edge)
    """
    # 1. Primary Assignment (Initial Guess)
    primary_assignment(tasks, num_cores, num_edge_nodes, num_edge_cores, upload_rates, download_rates)

    # 2. Task Prioritizing (Based on Critical Path)
    task_prioritizing(tasks)

    # 3. Execution Unit Selection (Refined Scheduling)
    scheduler = InitialTaskScheduler(tasks, num_cores, num_edge_nodes, num_edge_cores, upload_rates, download_rates)
    sequences = scheduler.run_initial_scheduling() # This performs the detailed scheduling

    # Final check on task attributes after scheduling
    for task in tasks:
        if task.assignment == -2: # Should have been assigned
            print(f"Warning: Task {task.id} remained unassigned after initial scheduling.")
        # Ensure tier matches assignment
        task.update_assignment_and_tier(task.assignment, num_cores, num_edge_nodes, num_edge_cores)


    return sequences

def construct_sequence(tasks, task_id, execution_unit, original_sequence,
                      num_cores, num_edge_nodes, num_edge_cores, # System params
                      upload_rates): # Network param needed for edge fix
    """
    Implements the linear-time rescheduling algorithm for three-tier architecture.
    Constructs new sequence after task migration while preserving task precedence.
    Includes the fix for edge migration ordering using estimated upload finish time.

    Args:
        tasks: List of all tasks in the application
        task_id: ID of task v_tar being migrated (1-based)
        execution_unit: New execution location k_tar (0-based index: 0..K-1 device, K cloud, K+1.. edge)
        original_sequence: Current Sk sequences for all execution units (list of lists)
        num_cores: Number of device cores (K)
        num_edge_nodes: Number of edge nodes (M)
        num_edge_cores: Number of cores per edge node
        upload_rates: Dictionary of upload bandwidths {link_type: rate_mbps}

    Returns:
        Modified sequence sets after migrating the task (list of lists)
    """
    task_id_to_task = {task.id: task for task in tasks} # O(N) or O(1) avg if dict
    target_task = task_id_to_task.get(task_id)
    if not target_task:
        print(f"Error: Task ID {task_id} not found.")
        return original_sequence

    # Define key indices based on actual core count
    cloud_index = num_cores
    edge_start_index = num_cores + 1

    # --- Step 1: Determine Tier and Appropriate Ready Time for Insertion ---
    # This RT determines the earliest the task *could* start its critical
    # activity on the new unit, dictating its sequence position.
    target_task_rt_for_bisect = 0.0 # The time value used for sorting/insertion
    target_tier = None

    if 0 <= execution_unit < num_cores: # Target is Device Core
        target_task_rt_for_bisect = getattr(target_task, 'RT_l', 0.0)
        target_tier = ExecutionTier.DEVICE
    elif execution_unit == cloud_index: # Target is Cloud
        target_task_rt_for_bisect = getattr(target_task, 'RT_ws', 0.0) # Use upload ready time
        target_tier = ExecutionTier.CLOUD
    elif execution_unit >= edge_start_index: # Target is Edge Unit
        target_tier = ExecutionTier.EDGE
        # Find the specific edge (e_id, c_id) for this execution_unit index
        if num_edge_cores > 0:
            edge_offset = execution_unit - edge_start_index
            e_id = (edge_offset // num_edge_cores) + 1
            c_id = (edge_offset % num_edge_cores) + 1

            # Use estimated UPLOAD FINISH time as the heuristic ready time for bisect
            # Requires RT_device_to_edge, data size, and upload rate.

            # Get the ready time to start the upload
            rt_dev_to_edge_upload_start = getattr(target_task, 'RT_device_to_edge', {}).get(e_id, 0.0)

            # Estimate upload time
            estimated_upload_time = 0.0
            data_key = f'device_to_edge{e_id}'
            rate_key = 'device_to_edge'
            if hasattr(target_task, 'data_sizes') and data_key in target_task.data_sizes:
                 up_size_mb = target_task.data_sizes[data_key]
                 # Use specific rate if available, else generic, else default
                 up_rate_mbps = upload_rates.get(f'device_to_edge{e_id}', upload_rates.get(rate_key, 1.0))
                 if up_rate_mbps > 0:
                     estimated_upload_time = (up_size_mb * 8.0) / up_rate_mbps

            # The heuristic RT is when the upload is estimated to finish
            target_task_rt_for_bisect = rt_dev_to_edge_upload_start + estimated_upload_time

        else:
            print(f"Error: execution_unit {execution_unit} implies edge, but num_edge_cores is 0.")
            # Fallback: Treat as device?
            target_task_rt_for_bisect = getattr(target_task, 'RT_l', 0.0)
            target_tier = ExecutionTier.DEVICE
            execution_unit = 0 # Assign to core 0
    else: # Invalid execution_unit index
        print(f"Error: Invalid target execution_unit {execution_unit} for task {task_id}")
        return original_sequence # Return original sequences unchanged


    # --- Step 2: Remove task from original sequence ---
    original_assignment = target_task.assignment
    if 0 <= original_assignment < len(original_sequence):
        try:
            original_sequence[original_assignment].remove(task_id)
        except ValueError:
            # Task might have already been moved or wasn't in the expected sequence
             # print(f"Warning: Task {task_id} not found in original sequence {original_assignment}.")
             pass # Continue, attempt insertion anyway
    #else: # Don't warn if assignment was already invalid (e.g., -2)
        # print(f"Warning: Original assignment {original_assignment} for task {task_id} is invalid.")


    # --- Step 3: Get sequence for new execution unit & prepare for insertion ---
    # Ensure the target sequence list exists and handle potential index out of bounds
    while execution_unit >= len(original_sequence):
        original_sequence.append([])
    new_sequence_task_list = original_sequence[execution_unit]

    # Get the relevant start times for tasks *currently* in the new sequence
    # The 'relevant' start time depends on the *target* tier
    start_times = []
    valid_task_ids_in_new_seq = []
    for existing_task_id in new_sequence_task_list:
         task = task_id_to_task.get(existing_task_id)
         relevant_start_time = -1.0 # Default if not found/applicable

         if task and hasattr(task, 'execution_unit_task_start_times') and 0 <= execution_unit < len(task.execution_unit_task_start_times):
             # Get the actual start time recorded for this unit by the scheduler
             actual_start_time = task.execution_unit_task_start_times[execution_unit]

             # Use the start time relevant to the *target* tier for comparison consistency
             if target_tier == ExecutionTier.DEVICE:
                 relevant_start_time = actual_start_time # Compare local exec start
             elif target_tier == ExecutionTier.CLOUD:
                 relevant_start_time = actual_start_time # Compare cloud upload start
             elif target_tier == ExecutionTier.EDGE:
                 # Compare against estimated upload finish time for edge tasks already in sequence
                 # This provides a consistent basis for comparison with the migrating task's estimated upload finish
                 rt_dev_to_edge_upload_start = getattr(task, 'RT_device_to_edge', {}).get(e_id, 0.0)
                 est_upload_time = 0.0
                 data_key = f'device_to_edge{e_id}'
                 rate_key = 'device_to_edge'
                 if hasattr(task, 'data_sizes') and data_key in task.data_sizes:
                     up_size_mb = task.data_sizes[data_key]
                     up_rate_mbps = upload_rates.get(f'device_to_edge{e_id}', upload_rates.get(rate_key, 1.0))
                     if up_rate_mbps > 0:
                         est_upload_time = (up_size_mb * 8.0) / up_rate_mbps
                 relevant_start_time = rt_dev_to_edge_upload_start + est_upload_time
             else: # Fallback
                 relevant_start_time = actual_start_time if actual_start_time >= 0 else float('inf')

         else: # Task data inconsistent or start time not set
             relevant_start_time = float('inf') # Place migrating task before unknowns

         start_times.append(relevant_start_time)
         valid_task_ids_in_new_seq.append(existing_task_id)

    # Ensure sequence list only contains valid tasks used for sorting
    original_sequence[execution_unit] = valid_task_ids_in_new_seq


    # --- Step 4: Find insertion point using binary search ---
    # Uses the calculated heuristic ready time for the migrating task
    # against the relevant heuristic start/readiness times of tasks in the target sequence
    insertion_index = bisect.bisect_left(start_times, target_task_rt_for_bisect)

    # --- Step 5: Insert task at correct position ---
    original_sequence[execution_unit].insert(insertion_index, task_id)

    # --- Step 6: Update task attributes based on new execution unit index ---
    target_task.update_assignment_and_tier(execution_unit, num_cores, num_edge_nodes, num_edge_cores)

    # --- Step 7: Reset scheduling state ---
    # Ensures the kernel algorithm will process this task fresh
    target_task.is_scheduled = SchedulingState.UNSCHEDULED

    return original_sequence

class KernelScheduler:
    """
    Implements the kernel scheduling algorithm for three-tier architecture.
    Provides linear-time task rescheduling while preserving dependencies.
    Handles resource tracking across device, edge, and cloud.
    """

    def __init__(self, tasks, sequences, num_cores, num_edge_nodes, num_edge_cores, # Added params
                 upload_rates, download_rates): # Added network
        """
        Initialize kernel scheduler with three-tier resource tracking.

        Args:
            tasks: List of Task objects to schedule
            sequences: Task sequences for each execution unit
            num_cores: Number of device cores
            num_edge_nodes: Number of edge nodes
            num_edge_cores: Number of cores per edge node
            upload_rates: Upload bandwidths
            download_rates: Download bandwidths
        """
        self.tasks = tasks
        self.sequences = sequences # Current sequences after potential migration
        self.num_cores = num_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores = num_edge_cores

        self.upload_rates = upload_rates
        self.download_rates = download_rates

        # Define indices
        self.cloud_index = num_cores
        self.edge_start_index = num_cores + 1
        self.num_edge_units = num_edge_nodes * num_edge_cores
        self.total_units = self.edge_start_index + self.num_edge_units

        # --- RESOURCE AVAILABILITY TRACKING (reset for kernel run) ---
        self.core_earliest_ready = [0.0] * self.num_cores
        self.cloud_upload_channel_ready = 0.0
        self.cloud_download_channel_ready = 0.0
        self.edge_cores_earliest_ready = {} # Map (e_id, c_id) -> time
        self.device_to_edge_channels_ready = {} # Map e_id -> time
        self.edge_to_device_channels_ready = {} # Map e_id -> time
        # Initialize edge trackers
        for e_id in range(1, num_edge_nodes + 1):
            self.device_to_edge_channels_ready[e_id] = 0.0
            self.edge_to_device_channels_ready[e_id] = 0.0
            for c_id in range(1, num_edge_cores + 1):
                self.edge_cores_earliest_ready[(e_id, c_id)] = 0.0

        # --- TASK STATE TRACKING (reset for kernel run) ---
        self.dependency_ready = {} # Map task_id -> count of unscheduled predecessors
        self.sequence_ready = {}   # Map task_id -> 0 (ready) or 1 (waiting in sequence)
        self.initialize_task_state()

        self.task_map = {task.id: task for task in tasks} # Quick lookup

    def get_edge_unit_index(self, edge_id, core_id):
        """Maps an edge (node_id, core_id) to its index in the sequences list."""
        if not (1 <= edge_id <= self.num_edge_nodes and 1 <= core_id <= self.num_edge_cores):
            return -1
        offset = (edge_id - 1) * self.num_edge_cores + (core_id - 1)
        return self.edge_start_index + offset

    def initialize_task_state(self):
        """Initializes ready1 (dependency) and ready2 (sequence) vectors."""
        for task in self.tasks:
            task.is_scheduled = SchedulingState.UNSCHEDULED # Reset schedule state
            self.dependency_ready[task.id] = len(task.pred_tasks)
            self.sequence_ready[task.id] = 1 # Assume waiting initially

        # Set sequence_ready to 0 for tasks at the start of each sequence
        for sequence in self.sequences:
            if sequence:
                first_task_id = sequence[0]
                self.sequence_ready[first_task_id] = 0

    def update_task_state_after_scheduling(self, scheduled_task_id):
        """Update ready vectors for successors after a task is scheduled."""
        scheduled_task = self.task_map.get(scheduled_task_id)
        if not scheduled_task: return

        # Decrement dependency count for all direct successors
        for succ_task in scheduled_task.succ_tasks:
            if succ_task.id in self.dependency_ready:
                self.dependency_ready[succ_task.id] -= 1
                self.dependency_ready[succ_task.id] = max(0, self.dependency_ready[succ_task.id]) # Ensure non-negative


        # Update sequence readiness for the *next* task in the *same* sequence
        current_assignment = scheduled_task.assignment
        if 0 <= current_assignment < len(self.sequences):
             sequence = self.sequences[current_assignment]
             try:
                 idx = sequence.index(scheduled_task_id)
                 if idx + 1 < len(sequence): # If not the last task in sequence
                     next_task_id = sequence[idx + 1]
                     if next_task_id in self.sequence_ready:
                         self.sequence_ready[next_task_id] = 0 # Mark next task as ready in sequence
             except ValueError:
                 pass # Task not found in sequence, ignore


    def schedule_local_task(self, task):
        """Schedules task on its assigned device core, calculating actual start/finish."""
        core_id = task.assignment
        if not (0 <= core_id < self.num_cores):
             print(f"Error: Task {task.id} assigned to invalid device core {core_id}")
             return

        # Calculate ready time (must wait for all predecessors)
        max_pred_finish = 0.0
        if task.pred_tasks:
             max_pred_finish = max(self._get_predecessor_finish_time_at_device(p) for p in task.pred_tasks)
        task.RT_l = max_pred_finish # Update RT based on current schedule

        # Calculate start time (max of ready time and core availability)
        start_time = max(task.RT_l, self.core_earliest_ready[core_id])

        # Get execution time
        exec_time = task.local_execution_times[core_id] if core_id < len(task.local_execution_times) else 0.0
        finish_time = start_time + exec_time

        # Update task timings
        task.execution_unit_task_start_times[core_id] = start_time
        task.FT_l = finish_time
        task.execution_finish_time = finish_time # Final time is local finish

        # Update resource availability
        self.core_earliest_ready[core_id] = finish_time


    def schedule_cloud_task(self, task):
        """Schedules task for cloud execution, calculating actual phase timings."""
        if not task.cloud_execution_times or len(task.cloud_execution_times) < 3:
            print(f"Error: Task {task.id} assigned to cloud but missing cloud times.")
            return

        t_send, t_compute, t_receive = task.cloud_execution_times[:3]

        # Calculate ready time for upload (pred finish at device + channel availability)
        max_pred_finish_at_device = 0.0
        if task.pred_tasks:
            max_pred_finish_at_device = max(self._get_predecessor_finish_time_at_device(p) for p in task.pred_tasks)
        task.RT_ws = max_pred_finish_at_device # Base ready time

        # Phase 1: Upload
        upload_start = max(task.RT_ws, self.cloud_upload_channel_ready)
        upload_finish = upload_start + t_send
        task.FT_ws = upload_finish
        self.cloud_upload_channel_ready = upload_finish

        # Phase 2: Compute
        # Ready for compute depends on upload finishing AND cloud predecessors finishing compute
        max_cloud_pred_compute_finish = 0.0
        if task.pred_tasks:
             cloud_preds = [p for p in task.pred_tasks if p.execution_tier == ExecutionTier.CLOUD]
             if cloud_preds:
                 max_cloud_pred_compute_finish = max(getattr(p, 'FT_c', 0.0) for p in cloud_preds)
        task.RT_c = max(upload_finish, max_cloud_pred_compute_finish) # Ready to compute
        compute_start = task.RT_c # Cloud assumed infinitely parallel
        compute_finish = compute_start + t_compute
        task.FT_c = compute_finish

        # Phase 3: Download
        # Ready for download when compute finishes
        task.RT_wr = compute_finish
        download_start = max(task.RT_wr, self.cloud_download_channel_ready)
        download_finish = download_start + t_receive
        task.FT_wr = download_finish
        self.cloud_download_channel_ready = download_finish

        # Update task timings
        task.execution_unit_task_start_times[self.cloud_index] = upload_start
        task.execution_finish_time = download_finish # Final time is download finish

    def schedule_edge_task(self, task):
        """Schedules task on its assigned edge core, calculating phase timings."""
        if not task.edge_assignment:
             print(f"Error: Task {task.id} assigned to edge but missing assignment details.")
             return

        e_id = task.edge_assignment.edge_id
        c_id = task.edge_assignment.core_id
        edge_core_key = (e_id, c_id)
        sequence_index = task.assignment # Should match get_edge_unit_index(e_id, c_id)

        exec_time = task.get_edge_execution_time(e_id, c_id)
        if exec_time == float('inf'):
             print(f"Error: Task {task.id} assigned to edge ({e_id},{c_id}) with infinite exec time.")
             return

        # Calculate ready time for upload (pred finish at device + channel availability)
        max_pred_finish_at_device = 0.0
        if task.pred_tasks:
            max_pred_finish_at_device = max(self._get_predecessor_finish_time_at_device(p) for p in task.pred_tasks)

        # Update RT for device->edge upload (distinct from RT_edge for compute)
        task.RT_device_to_edge[e_id] = max_pred_finish_at_device

        # Phase 1: Upload (Device -> Edge)
        up_data_key = f'device_to_edge{e_id}'
        up_rate_key = 'device_to_edge'
        up_size_mb = task.data_sizes.get(up_data_key, 0.0)
        up_rate_mbps = self.upload_rates.get(f'device_to_edge{e_id}', self.upload_rates.get(up_rate_key, 1.0))
        upload_time = (up_size_mb * 8.0) / up_rate_mbps if up_rate_mbps > 0 else 0.0

        upload_start = max(task.RT_device_to_edge[e_id], self.device_to_edge_channels_ready.get(e_id, 0.0))
        upload_finish = upload_start + upload_time
        self.device_to_edge_channels_ready[e_id] = upload_finish

        # Phase 2: Edge Execution
        # Ready depends on upload AND potentially predecessors on same edge finishing
        max_same_edge_pred_exec_finish = 0.0
        if task.pred_tasks:
            same_edge_preds = [p for p in task.pred_tasks
                               if p.edge_assignment and p.edge_assignment.edge_id == e_id]
            if same_edge_preds:
                 max_same_edge_pred_exec_finish = max(
                     getattr(p, 'FT_edge', {}).get((e_id, p.edge_assignment.core_id), 0.0)
                     for p in same_edge_preds if p.edge_assignment
                 )

        task.RT_edge[edge_core_key] = max(upload_finish, max_same_edge_pred_exec_finish) # Ready to compute on edge
        exec_start = max(task.RT_edge[edge_core_key], self.edge_cores_earliest_ready.get(edge_core_key, 0.0))
        exec_finish = exec_start + exec_time
        task.FT_edge[edge_core_key] = exec_finish
        self.edge_cores_earliest_ready[edge_core_key] = exec_finish

        # Phase 3: Download (Edge -> Device)
        down_data_key = f'edge{e_id}_to_device'
        down_rate_key = 'edge_to_device'
        down_size_mb = task.data_sizes.get(down_data_key, 0.0)
        down_rate_mbps = self.download_rates.get(f'edge{e_id}_to_device', self.download_rates.get(down_rate_key, 1.0))
        download_time = (down_size_mb * 8.0) / down_rate_mbps if down_rate_mbps > 0 else 0.0

        download_ready = exec_finish # Ready when edge exec finishes
        download_start = max(download_ready, self.edge_to_device_channels_ready.get(e_id, 0.0))
        download_finish = download_start + download_time
        task.FT_edge_receive[e_id] = download_finish
        self.edge_to_device_channels_ready[e_id] = download_finish

        # Update task timings
        if 0 <= sequence_index < len(task.execution_unit_task_start_times):
            task.execution_unit_task_start_times[sequence_index] = exec_start
        task.execution_finish_time = download_finish # Final time is download finish


    def initialize_queue(self):
        """Initializes LIFO stack with tasks ready based on dependency and sequence."""
        ready_queue = deque()
        for task_id, dep_count in self.dependency_ready.items():
            # Ready if no unscheduled predecessors AND ready in sequence
            if dep_count == 0 and self.sequence_ready.get(task_id) == 0:
                 task = self.task_map.get(task_id)
                 # Add only if not already scheduled in this kernel run
                 if task and task.is_scheduled == SchedulingState.UNSCHEDULED:
                     ready_queue.append(task)
        return ready_queue

    def _get_predecessor_finish_time_at_device(self, pred_task):
        """Helper identical to the one in InitialTaskScheduler."""
        if not pred_task: return 0.0
        return pred_task.get_final_finish_time()


def kernel_algorithm(tasks, sequences, num_cores, num_edge_nodes, num_edge_cores, upload_rates, download_rates):
    """
    Implements linear-time rescheduling algorithm for three-tier architecture.

    Args:
        tasks: List of Task objects
        sequences: Task sequences for each execution unit
        num_cores, num_edge_nodes, num_edge_cores: System parameters
        upload_rates, download_rates: Network parameters

    Returns:
        Updated tasks with new scheduling information across all three tiers
    """
    # Initialize kernel scheduler
    scheduler = KernelScheduler(tasks, sequences, num_cores, num_edge_nodes, num_edge_cores,
                                upload_rates, download_rates)

    # Initialize LIFO stack with ready tasks
    queue = scheduler.initialize_queue()

    processed_count = 0
    max_processed = len(tasks) * 2 # Safety break

    # Main scheduling loop
    while queue and processed_count < max_processed:
        current_task = queue.popleft() # Use popleft for FIFO behavior if desired (doesn't affect O(N))
        processed_count += 1

        # Skip if already processed in this run (can happen if added multiple times)
        if current_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED:
            continue

        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on assigned execution tier
        tier = current_task.execution_tier
        if tier == ExecutionTier.DEVICE:
            scheduler.schedule_local_task(current_task)
        elif tier == ExecutionTier.CLOUD:
            scheduler.schedule_cloud_task(current_task)
        elif tier == ExecutionTier.EDGE:
            scheduler.schedule_edge_task(current_task)
        else:
            print(f"Error: Task {current_task.id} has invalid tier {tier} during kernel algorithm.")
            continue # Skip task

        # Update dependency and sequence readiness for successors
        scheduler.update_task_state_after_scheduling(current_task.id)

        # Add newly ready tasks to queue
        # Check successors of the just-scheduled task
        for succ_task in current_task.succ_tasks:
             succ_id = succ_task.id
             # Check if successor is now ready (dependencies met AND sequence ready)
             if (scheduler.dependency_ready.get(succ_id) == 0 and
                 scheduler.sequence_ready.get(succ_id) == 0 and
                 succ_task.is_scheduled == SchedulingState.UNSCHEDULED):
                  # Add to queue only if not already present
                  if succ_task not in queue:
                      queue.append(succ_task)
        # Also check the next task in the sequence (if different from successors)
        current_assignment = current_task.assignment
        if 0 <= current_assignment < len(sequences):
            sequence = sequences[current_assignment]
            try:
                idx = sequence.index(current_task.id)
                if idx + 1 < len(sequence):
                    next_task_id = sequence[idx + 1]
                    next_task = scheduler.task_map.get(next_task_id)
                    if next_task and next_task.is_scheduled == SchedulingState.UNSCHEDULED:
                         # Check if sequence-successor is now ready
                         if (scheduler.dependency_ready.get(next_task_id) == 0 and
                             scheduler.sequence_ready.get(next_task_id) == 0):
                              if next_task not in queue:
                                  queue.append(next_task)
            except ValueError:
                 pass


    if processed_count >= max_processed:
        print("Error: Kernel algorithm exceeded max iterations. Possible cycle or logic error.")

    # Reset scheduling state for potential next migration iteration
    # (optimize_schedule handles the main loop; kernel is just one reschedule)
    # for task in tasks:
    #     task.is_scheduled = SchedulingState.UNSCHEDULED # Reset done by optimize_schedule loop

    return tasks

def generate_cache_key(tasks, task_idx, target_execution_unit):
    """
    Generates cache key for memoizing migration evaluations for three-tier architecture.
    Includes task index, target unit, and snapshot of all task assignments.
    """
    # Capture assignment state including tier/specific location
    assignment_state = []
    for task in tasks:
         # Use a tuple representing the assignment state
         state_tuple = (task.assignment, task.execution_tier,
                        task.device_core if task.device_core != -1 else None,
                        (task.edge_assignment.edge_id, task.edge_assignment.core_id) if task.edge_assignment else None)
         assignment_state.append(state_tuple)

    return (task_idx, target_execution_unit, tuple(assignment_state))

def evaluate_migration(tasks, seqs, task_idx, target_execution_unit, migration_cache,
                      device_power_profiles, rf_power, upload_rates, download_rates, # Added download_rates
                      num_cores, num_edge_nodes, num_edge_cores): # Added params
    """
    Evaluates potential task migration scenario for three-tier architecture.
    Uses caching, rescheduling, and recalculates time/energy.
    """
    # Generate cache key
    cache_key = generate_cache_key(tasks, task_idx, target_execution_unit)
    if cache_key in migration_cache:
        return migration_cache[cache_key]

    # --- Simulation ---
    # Create copies to simulate migration without affecting original state
    sequence_copy = [seq.copy() for seq in seqs] # Deep copy of sequence lists
    tasks_copy = deepcopy(tasks) # Deep copy of task objects

    # 1. Modify sequence based on migration
    sequence_copy = construct_sequence(
        tasks_copy,
        task_idx + 1,  # Task ID is 1-based
        target_execution_unit, # 0-based index
        sequence_copy,
        num_cores, num_edge_nodes, num_edge_cores, upload_rates # Pass params
    )

    # 2. Run kernel algorithm to reschedule based on new sequence
    # Pass all necessary parameters for the scheduler
    tasks_rescheduled = kernel_algorithm(
        tasks_copy, sequence_copy,
        num_cores, num_edge_nodes, num_edge_cores,
        upload_rates, download_rates
    )

    # 3. Calculate new metrics after rescheduling
    migration_T = total_time(tasks_rescheduled)
    migration_E = total_energy(tasks_rescheduled, device_power_profiles, rf_power, upload_rates)

    # Cache result
    migration_cache[cache_key] = (migration_T, migration_E)
    return migration_T, migration_E


def initialize_migration_choices(tasks, num_cores, num_edge_nodes, num_edge_cores):
    """
    Initializes possible migration choices matrix for three-tier architecture.
    A task can potentially migrate to any other valid execution unit.
    """
    num_edge_units = num_edge_nodes * num_edge_cores
    total_units = num_cores + 1 + num_edge_units

    # Matrix: N tasks x total_units execution units. True means 'possible target'.
    migration_choices = np.ones((len(tasks), total_units), dtype=bool)

    # Rule out migrating a task to its *current* location
    for i, task in enumerate(tasks):
        current_unit_index = task.assignment
        if 0 <= current_unit_index < total_units:
            migration_choices[i, current_unit_index] = False
        # else: Task might be unassigned (-2), leave all targets as True initially

    # Add constraints? E.g., maybe migrating edge->edge is disallowed?
    # For now, assume any valid unit is a potential target, except current one.

    return migration_choices

def identify_optimal_migration(migration_trials_results, T_current, E_current, T_max):
    """
    Identifies optimal task migration for three-tier architecture based on paper's criteria.
    Handles results containing device, edge, and cloud units.

    Args:
        migration_trials_results: List of tuples: (task_idx, target_unit_idx, time_after, energy_after)
        T_current: Current application completion time before migration
        E_current: Current energy consumption before migration
        T_max: Maximum allowed completion time constraint

    Returns:
        TaskMigrationState object with details of the selected migration, or None.
    """
    best_direct_reduction = {'reduction': -1, 'migration': None}
    tradeoff_candidates = [] # Min-heap storing (-efficiency, task_idx, unit_idx, time, energy)

    for task_idx, unit_idx, time_after, energy_after in migration_trials_results:
        # Ensure migration respects time constraint
        if time_after > T_max:
            continue

        energy_reduction = E_current - energy_after

        # Criterion 1: Largest energy reduction with NO time increase
        if time_after <= T_current and energy_reduction > 0:
             if energy_reduction > best_direct_reduction['reduction']:
                 best_direct_reduction['reduction'] = energy_reduction
                 best_direct_reduction['migration'] = (task_idx, unit_idx, time_after, energy_after)

        # Criterion 2: Best energy reduction / time increase ratio
        elif energy_reduction > 0: # Only consider migrations that save energy
             time_increase = time_after - T_current # Already know time_after > T_current here
             if time_increase <= 0: # Should have been caught by Criterion 1, but handle defensively
                 efficiency = float('inf')
             else:
                 efficiency = energy_reduction / time_increase

             # Use negative efficiency for max-heap behavior with heapq
             heappush(tradeoff_candidates, (-efficiency, task_idx, unit_idx, time_after, energy_after))


    # Prioritize direct reduction (Criterion 1)
    if best_direct_reduction['migration']:
        task_idx, unit_idx, time, energy = best_direct_reduction['migration']
        return TaskMigrationState(
            time=time,
            energy=energy,
            efficiency=best_direct_reduction['reduction'], # For direct, efficiency isn't ratio
            task_index=task_idx + 1,  # Convert back to 1-based ID
            target_execution_unit=unit_idx # Keep 0-based index
        )

    # If no direct reduction, use best tradeoff (Criterion 2)
    elif tradeoff_candidates:
        neg_efficiency, task_idx, unit_idx, time, energy = heappop(tradeoff_candidates)
        return TaskMigrationState(
            time=time,
            energy=energy,
            efficiency=-neg_efficiency, # Restore positive efficiency ratio
            task_index=task_idx + 1, # Convert back to 1-based ID
            target_execution_unit=unit_idx # Keep 0-based index
        )

    # No beneficial migration found
    return None

def optimize_schedule(tasks, sequence, T_max,
                     device_power_profiles, rf_power, upload_rates, download_rates,
                     num_cores, num_edge_nodes, num_edge_cores):
    """
    Implements the iterative task migration algorithm for the three-tier architecture.
    Optimizes energy consumption while respecting completion time constraints (T_max).
    Uses a boolean flag for loop control based on energy improvement, stopping when
    no further energy reduction is found.

    Args:
        tasks: List of Task objects from application graph G=(V,E)
        sequence: Initial Sk sequences from minimal-delay scheduling (list of lists)
        T_max: Maximum allowed application completion time constraint
        device_power_profiles: Power models for device cores (dict)
        rf_power: RF component power models (dict)
        upload_rates: Upload bandwidths (dict)
        download_rates: Download bandwidths (dict)
        num_cores: Number of device cores (int)
        num_edge_nodes: Number of edge nodes (int)
        num_edge_cores: Number of cores per edge node (int)

    Returns:
        tuple: (tasks, sequence, migrations)
            - tasks: List of Task objects with optimized scheduling.
            - sequence: List of lists representing the final task sequences Sk.
            - migrations: List of dictionaries detailing the applied migrations.
    """
    # System configuration parameters needed for helper functions
    num_edge_units = num_edge_nodes * num_edge_cores
    total_units = num_cores + 1 + num_edge_units # Device Cores + Cloud + Edge Units

    # Cache for memoizing evaluate_migration results
    migration_cache = {}
    # History of applied migrations
    migrations = []

    # Calculate initial metrics before optimization starts
    current_energy = total_energy(tasks, device_power_profiles, rf_power, upload_rates)
    current_time = total_time(tasks)
    print(f"Optimize Start: T={current_time:.2f}, E={current_energy:.2f}, T_max={T_max:.2f}")

    # --- Iterative Improvement Loop using Boolean Flag ---
    energy_improved = True # Assume improvement is possible initially
    iteration = 0 # Track number of iterations

    while energy_improved:
        iteration += 1
        print(f"\nMigration Iteration {iteration}")
        energy_before_iter = current_energy
        time_before_iter = current_time
        energy_improved = False # Reset flag: assume no improvement this iteration

        # Initialize migration possibilities matrix (which targets are valid to try)
        # Excludes migrating to the task's current location
        migration_options = initialize_migration_choices(tasks, num_cores, num_edge_nodes, num_edge_cores)

        # Evaluate all valid potential migrations
        migration_trials_results = []
        num_evaluated = 0
        for task_idx in range(len(tasks)): # 0-based index
            for target_unit_idx in range(total_units): # 0-based index
                 # Skip if this is not a valid target to try (e.g., migrating to self)
                 if not migration_options[task_idx, target_unit_idx]:
                     continue

                 # Evaluate the outcome of this potential migration
                 mig_time, mig_energy = evaluate_migration(
                     tasks, sequence, task_idx, target_unit_idx, migration_cache,
                     device_power_profiles, rf_power, upload_rates, download_rates,
                     num_cores, num_edge_nodes, num_edge_cores
                 )
                 num_evaluated += 1
                 migration_trials_results.append(
                     (task_idx, target_unit_idx, mig_time, mig_energy)
                 )

        print(f" Evaluated {num_evaluated} potential migrations.")

        # Select the best migration based on energy/time criteria defined in the paper
        best_migration_state = identify_optimal_migration(
            migration_trials_results=migration_trials_results,
            T_current=time_before_iter,
            E_current=energy_before_iter,
            T_max=T_max
        )

        # --- Apply Migration and Update State (If Found & Beneficial) ---
        if best_migration_state is not None:
            # Check if the selected migration *actually* reduces energy
            # Use a small tolerance for floating point comparison if necessary
            if best_migration_state.energy < energy_before_iter: # (energy_before_iter - best_migration_state.energy) > epsilon:
                energy_improved = True # Improvement found, set flag to continue loop

                target_task_id = best_migration_state.task_index # 1-based ID from state object
                target_unit_idx = best_migration_state.target_execution_unit # 0-based index
                # Ensure task_id is valid before indexing
                if not (1 <= target_task_id <= len(tasks)):
                    print(f"  ERROR: Invalid task index {target_task_id} in migration state. Stopping.")
                    energy_improved = False # Prevent further loops
                    continue

                task_obj = tasks[target_task_id - 1] # Get the actual task object

                # Record migration details before applying
                from_unit_idx = task_obj.assignment
                from_tier = task_obj.execution_tier
                # Determine target tier based on index
                if 0 <= target_unit_idx < num_cores: to_tier = ExecutionTier.DEVICE
                elif target_unit_idx == num_cores: to_tier = ExecutionTier.CLOUD
                else: to_tier = ExecutionTier.EDGE

                migrations.append({
                    'iteration': iteration,
                    'task_id': target_task_id,
                    'from_unit_idx': from_unit_idx,
                    'from_tier': from_tier.name if from_tier else 'Unknown', # Use .name for enum
                    'to_unit_idx': target_unit_idx,
                    'to_tier': to_tier.name, # Use .name for enum
                    'time_before': time_before_iter, 'time_after': best_migration_state.time,
                    'energy_before': energy_before_iter, 'energy_after': best_migration_state.energy,
                    'efficiency': best_migration_state.efficiency
                })
                print(f" Selected Migration: Task {target_task_id} from unit {from_unit_idx} ({from_tier.name if from_tier else 'N/A'}) "
                      f"to unit {target_unit_idx} ({to_tier.name}). "
                      f"T: {time_before_iter:.3f}->{best_migration_state.time:.3f}, "
                      f"E: {energy_before_iter:.3f}->{best_migration_state.energy:.3f}")

                # 1. Update sequences
                sequence = construct_sequence(
                    tasks,
                    target_task_id,
                    target_unit_idx,
                    sequence, # Pass current sequences
                    num_cores, num_edge_nodes, num_edge_cores, upload_rates
                )

                # 2. Run kernel algorithm to reschedule *all* tasks based on new sequences
                # This updates the FT/RT times within the 'tasks' list objects
                tasks = kernel_algorithm(
                    tasks, sequence,
                    num_cores, num_edge_nodes, num_edge_cores,
                    upload_rates, download_rates
                )

                # 3. Update current time and energy for the *next* iteration's comparison
                current_time = total_time(tasks)
                current_energy = total_energy(tasks, device_power_profiles, rf_power, upload_rates)

                # Optional Cache Cleanup
                # Adjust cache size limit as needed
                if len(migration_cache) > 2000:
                    print("  Clearing migration cache.")
                    migration_cache.clear()

            else:
                 # Best identified move didn't strictly reduce energy. Stop.
                 print(f" Selected migration to E={best_migration_state.energy:.3f} does not strictly reduce energy from E={energy_before_iter:.3f}. Stopping.")
                 energy_improved = False # Ensure loop terminates

        else:
            # No beneficial migration found by identify_optimal_migration this iteration.
            print(" No beneficial migration found this iteration. Optimization finished.")
            energy_improved = False # Ensure loop terminates

    # --- End of Loop ---

    # Final energy/time calculated after the loop finishes
    final_T = total_time(tasks)
    final_E = total_energy(tasks, device_power_profiles, rf_power, upload_rates)
    print(f"\nOptimize End after {iteration} iterations: T={final_T:.2f}, E={final_E:.2f}")

    return tasks, sequence, migrations

def apply_configuration_parameters(config: MCCConfiguration):
    """
    Applies the configuration, generates models, and returns parameters.
    This version relies on config.apply() which handles most generation.
    It mainly ensures globals are updated correctly if needed elsewhere.
    """
    # Generate models using the configuration object's method
    params = config.apply()

    # Update globals based on what MCCConfiguration.apply generated
    # These globals are used by some functions (though passing params is safer)
    global core_execution_times, cloud_execution_times, edge_execution_times
    core_execution_times = params.get('core_execution_times', {})
    cloud_execution_times = params.get('cloud_execution_times', {})
    edge_execution_times = params.get('edge_execution_times', {}) # Will be populated later

    return params


def assign_task_attributes(tasks, config: MCCConfiguration):
    """
    Wrapper to add attributes like data sizes, complexity based on config.
    """
    return add_task_attributes(
        predefined_tasks=tasks,
        num_edge_nodes=config.num_edge_nodes,
        data_size_range=config.data_size_range,
        task_type_weights=config.task_type_distribution,
        # complexity_range=(0.5, 5.0), # Use defaults in add_task_attributes
        # data_intensity_range=(0.2, 2.0), # Use defaults
        scale_data_by_type=True,
        seed=config.seed
    )


def run_unified_test_3_tier(config: MCCConfiguration):
    """
    Run a full test (initial schedule + optimization) for a given configuration,
    supporting both two-tier and three-tier architectures.

    Args:
        config: MCCConfiguration object.

    Returns:
        dict: Test results and metrics, including initial and final task states.
    """
    start_run_time = time.time()
    print("-" * 60)
    print(f"Running Test: {config.name}")
    print(config) # Print config details

    # 1. Generate simulation parameters based on config
    params = apply_configuration_parameters(config)

    # 2. Create task graph structure
    tasks = generate_task_graph(
        num_tasks=40,
        complexity_level="high", 
        num_cores=config.num_cores,
        num_edge_nodes=config.num_edge_nodes,
        num_edge_cores=config.num_edge_cores,
        core_times=params['core_execution_times'],
        cloud_times=params['cloud_execution_times'],
        edge_times={},
        seed=config.seed
    )

    # 3. Assign detailed task attributes
    tasks = assign_task_attributes(tasks, config)

    # 4. Generate Edge Execution Times (if edge enabled)
    if config.num_edge_nodes > 0 and config.num_edge_cores > 0:
        print(" Generating edge execution times...")
        generate_edge_task_execution_times(
            tasks=tasks,
            mcc_edge_power_models=params['power_models'].get('edge', {}),
            num_edge_nodes=config.num_edge_nodes,
            num_edge_cores=config.num_edge_cores,
            seed=config.seed
        )
        global edge_execution_times
        edge_execution_times = {t.id: t.edge_execution_times for t in tasks if hasattr(t, 'edge_execution_times')}
        params['edge_execution_times'] = edge_execution_times

    # Extract parameters needed for scheduling functions
    upload_rates = params['upload_rates']
    download_rates = params['download_rates']
    power_models = params['power_models']
    device_power_profiles = power_models.get('device', {})
    rf_power = power_models.get('rf', {})

    # --- Initial Scheduling (Minimal Delay) ---
    print(" Performing initial scheduling (minimal delay)...")
    initial_start_time = time.time()
    # *** Use a copy for initial scheduling to preserve original task objects if needed ***
    tasks_for_initial_scheduling = deepcopy(tasks)
    sequence_initial = execution_unit_selection(
        tasks_for_initial_scheduling, # Use the copy
        num_cores=config.num_cores,
        num_edge_nodes=config.num_edge_nodes,
        num_edge_cores=config.num_edge_cores,
        upload_rates=upload_rates,
        download_rates=download_rates
    )
    initial_schedule_time = time.time() - initial_start_time

    # *** Store the state of tasks *after* initial scheduling ***
    tasks_initial_state = deepcopy(tasks_for_initial_scheduling)

    # Calculate metrics for initial schedule (using the scheduled task state)
    T_initial = total_time(tasks_initial_state)
    E_initial = total_energy(tasks_initial_state, device_power_profiles, rf_power, upload_rates)
    print(f" Initial Schedule: T={T_initial:.2f}, E={E_initial:.2f} (took {initial_schedule_time:.2f}s)")

    initial_dist = {tier: 0 for tier in ExecutionTier}
    for task in tasks_initial_state: initial_dist[task.execution_tier] += 1
    print(f"  Initial Distribution: {initial_dist}")

    is_valid_initial, violations_initial = validate_task_dependencies(tasks_initial_state)
    if not is_valid_initial:
        print("\nWARNING: Initial schedule has dependency violations!")

    # --- Energy Optimization (Task Migration) ---
    T_max = T_initial * config.time_constraint_multiplier
    print(f" Optimizing schedule for energy (T_max = {T_max:.2f})...")
    optimize_start_time = time.time()

    # *** Use the *initial scheduled state* as the starting point for optimization ***
    tasks_to_optimize = deepcopy(tasks_initial_state)
    sequence_to_optimize = [seq.copy() for seq in sequence_initial]

    tasks_final, sequence_final, migrations = optimize_schedule(
        tasks_to_optimize, sequence_to_optimize, T_max,
        device_power_profiles, rf_power, upload_rates, download_rates,
        config.num_cores, config.num_edge_nodes, config.num_edge_cores
    )
    optimize_schedule_time = time.time() - optimize_start_time

    # Calculate metrics for final schedule
    T_final = total_time(tasks_final)
    E_final = total_energy(tasks_final, device_power_profiles, rf_power, upload_rates)
    print(f" Optimized Schedule: T={T_final:.2f}, E={E_final:.2f} (took {optimize_schedule_time:.2f}s)")

    final_dist = {tier: 0 for tier in ExecutionTier}
    for task in tasks_final: final_dist[task.execution_tier] += 1
    print(f"  Final Distribution: {final_dist}")
    print(f"  Migrations: {len(migrations)}")

    is_valid_final, violations_final = validate_task_dependencies(tasks_final)
    if not is_valid_final:
        print("\nWARNING: Optimized schedule has dependency violations!")

    run_duration = time.time() - start_run_time
    print(f" Test run completed in {run_duration:.2f} seconds.")
    print("-" * 60)

    # Return comprehensive results, including both task states AND final sequence
    result_data = {
        'config': config,
        'config_name': config.name,
        'config_details': str(config),
        'num_cores': config.num_cores,
        'num_edge_nodes': config.num_edge_nodes,
        'num_edge_cores': config.num_edge_cores,
        'initial_time': T_initial,
        'final_time': T_final,
        'time_constraint': T_max,
        'initial_energy': E_initial,
        'final_energy': E_final,
        'time_change_percent': (T_final - T_initial) / T_initial * 100 if T_initial > 0 else 0,
        'energy_reduction_percent': (E_initial - E_final) / E_initial * 100 if E_initial > 0 else 0,
        'initial_distribution': initial_dist,
        'final_distribution': final_dist,
        'migration_count': len(migrations),
        'migrations': migrations,
        'initial_schedule_valid': is_valid_initial,
        'final_schedule_valid': is_valid_final,
        'initial_schedule_violations': violations_initial,
        'final_schedule_violations': violations_final,
        'initial_scheduling_duration': initial_schedule_time,
        'optimization_duration': optimize_schedule_time,
        'total_duration': run_duration,
        'tasks_initial_state': tasks_initial_state,
        'tasks_final_state': tasks_final,
        'sequence_initial': sequence_initial,
        'sequence_final': sequence_final,
        'params_summary': { # Keep relevant params summary
             'upload_rate_cloud': params.get('upload_rates', {}).get('device_to_cloud'),
             'download_rate_cloud': params.get('download_rates', {}).get('cloud_to_device'),
             'core_exec_times_sample': params.get('core_execution_times', {}).get(1),
             'cloud_exec_times_sample': params.get('cloud_execution_times', {}).get(1),
             'edge_exec_times_sample': params.get('edge_execution_times', {}).get(1) if params.get('edge_execution_times') else None,
         }
    }
    initial_local_count = initial_dist.get(ExecutionTier.DEVICE, 0)
    initial_edge_count = initial_dist.get(ExecutionTier.EDGE, 0)
    initial_cloud_count = initial_dist.get(ExecutionTier.CLOUD, 0)
    final_local_count = final_dist.get(ExecutionTier.DEVICE, 0)
    final_edge_count = final_dist.get(ExecutionTier.EDGE, 0)
    final_cloud_count = final_dist.get(ExecutionTier.CLOUD, 0)

    # Add counts to results
    result_data['initial_local_count'] = initial_local_count
    result_data['initial_edge_count'] = initial_edge_count
    result_data['initial_cloud_count'] = initial_cloud_count
    result_data['final_local_count'] = final_local_count
    result_data['final_edge_count'] = final_edge_count
    result_data['final_cloud_count'] = final_cloud_count

    # Calculate and add migrations for Edge and Cloud
    result_data['edge_migration'] = final_edge_count - initial_edge_count
    result_data['cloud_migration'] = final_cloud_count - initial_cloud_count
    result_data['deadline_met'] = (result_data['final_time'] <= result_data['time_constraint'])
    # --- End of Added Block ---

    return result_data

from data import MCCConfiguration, generate_configs, generate_single_random_config
import traceback

if __name__ == "__main__":

    # --- Control Flags ---
    run_specialized = True      # Run the predefined specific scenarios
    run_random = True           # Run randomly generated scenarios
    print_schedules = True      # Print detailed schedule comparison after each run
    num_random_tests = 3        # How many random configurations to generate and test

    # --- Lists to store results ---
    all_results = []
    all_failures = []
    configs_run_count = 0

    # --- NO Q-Learning Parameters Needed for this script ---

    # --- Section 1: Run Specific Specialized Tests ---
    if run_specialized:
        target_config_names = [
            "Local-Favoring_Cores_3",
            "Cloud-Favoring_BW_2.0",
            "Battery-Critical_15pct",
            "Three-Tier_Base",
            "Network-Constrained_Edge",
            "Heterogeneous_Edge"
        ]
        print("-" * 20 + " Running SPECIFIC Specialized Configurations (Heuristic Algo) " + "-" * 20)
        print(f"Target names: {target_config_names}")

        # Generate the pool of specialized configurations
        try:
            all_specialized_configs = generate_configs(param_ranges=None, seed=42)
            print(f"Found {len(all_specialized_configs)} predefined specialized configurations in total.")
        except Exception as e:
            print(f"FATAL ERROR generating specialized configurations: {e}")
            all_specialized_configs = []

        # Filter for the target configurations
        configs_to_test_specialized = [cfg for cfg in all_specialized_configs if cfg.name in target_config_names]

        # Validate selection
        if not configs_to_test_specialized:
            print("\n*** WARNING: No target specialized configurations found! Check names. Skipping specialized tests. ***\n")
        elif len(configs_to_test_specialized) < len(target_config_names):
             found_names = {cfg.name for cfg in configs_to_test_specialized}
             missing_names = [name for name in target_config_names if name not in found_names]
             print(f"\n*** WARNING: Could not find specialized configurations: {missing_names}. ***")
             print(f"*** Proceeding with {len(configs_to_test_specialized)} found specialized configurations. ***\n")
        else:
             print(f"Successfully selected all {len(configs_to_test_specialized)} target specialized configurations.")

        # Execute tests if configurations were found
        if configs_to_test_specialized:
            print(f"\n--- Starting execution of {len(configs_to_test_specialized)} specialized tests ---")

            for i, config in enumerate(configs_to_test_specialized):
                test_name = f"Specialized Test {i+1}/{len(configs_to_test_specialized)}"
                config_id_name = config.name
                print(f"\n--- {test_name}: Running '{config_id_name}' ---")
                configs_run_count += 1
                start_single_test = time.time()
                result = None

                try:
                    # --- Run the core test function for the heuristic algo ---
                    result = run_unified_test_3_tier(config=config) # No QL params needed

                    # --- Process result ---
                    if result and 'error' not in result:
                        all_results.append(result)
                        end_single_test = time.time()
                        print(f"--- Test '{config_id_name}' Completed Successfully in {end_single_test - start_single_test:.2f}s ---")
                    else:
                         error_msg = result.get('error', 'Unknown error structure returned.') if result else 'Function returned None.'
                         print(f"!!! Test '{config_id_name}' Failed during execution: {error_msg.splitlines()[0]} !!!")
                         all_failures.append({'config_name': config_id_name, 'error': error_msg})

                except Exception as e: # Catch unexpected errors
                    end_single_test = time.time()
                    error_msg = f"Unhandled Exception: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    print(f"!!! Test '{config_id_name}' Failed with Unhandled Exception after {end_single_test - start_single_test:.2f}s: {type(e).__name__} !!!")
                    all_failures.append({'config_name': config_id_name, 'error': error_msg})

    # --- Section 2: Run Randomly Generated Configurations ---
    if run_random:
        print("\n" + "-" * 20 + " Running RANDOMLY Generated Configurations (Heuristic Algo) " + "-" * 20)

        base_random_seed = 123 # Use a base seed for the sequence

        print(f"--- Starting execution of {num_random_tests} random tests ---")
        for i in range(num_random_tests):
            test_name = f"Random Test {i+1}/{num_random_tests}"
            print(f"\n--- {test_name} ---")
            configs_run_count += 1
            start_single_test = time.time()
            random_config = None
            result = None

            try:
                # Generate ONE random config
                random_config_seed = base_random_seed + i if base_random_seed is not None else None
                random_config = generate_single_random_config(
                    name_prefix="RandomTest",
                    base_seed=random_config_seed
                )
                config_id_name = random_config.name
                print(f"Generated Config: {config_id_name}")

                # --- Run the core test function for the heuristic algo ---
                result = run_unified_test_3_tier(config=random_config) # No QL params

                # --- Process result ---
                if result and 'error' not in result:
                    all_results.append(result)
                    end_single_test = time.time()
                    print(f"--- Test '{config_id_name}' Completed Successfully in {end_single_test - start_single_test:.2f}s ---")
                else:
                    error_msg = result.get('error', 'Unknown error structure returned.') if result else 'Function returned None.'
                    print(f"!!! Test '{config_id_name}' Failed during execution: {error_msg.splitlines()[0]} !!!")
                    all_failures.append({'config_name': config_id_name, 'error': error_msg})

            except Exception as e: # Catch unexpected errors
                end_single_test = time.time()
                config_name_fallback = random_config.name if random_config else f"RandomTest_{i+1}_GenFail"
                error_msg = f"Unhandled Exception: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"!!! Test '{config_name_fallback}' Failed with Unhandled Exception after {end_single_test - start_single_test:.2f}s: {type(e).__name__} !!!")
                all_failures.append({'config_name': config_name_fallback, 'error': error_msg})

    # --- Combined Summary of ALL Results ---
    print("\n" + "=" * 35 + " Overall Heuristic Test Run Summary " + "=" * 35) # Updated Title
    print(f"Total configurations attempted/run: {configs_run_count}")
    print(f"Successful tests: {len(all_results)}")
    print(f"Failed tests: {len(all_failures)}")

    if all_results:
        print("\nKey Metrics from Successful Runs:")
        # Adjusted header - using 'Migr' for heuristic migration count
        print("-" * 98)
        print(f"{'Config Name':<30} | {'T_init':>8} | {'T_final':>8} | {'T_max':>8} | {'E_init':>10} | {'E_final':>10} | {'Valid':>5} | {'Migr':>4} | {'Reduc%':>6}")
        print("-" * 98)
        for r in all_results:
            if r and 'config_name' in r and 'error' not in r:
                valid_str = "OK" if r.get('final_schedule_valid', False) else "FAIL"
                reduc_perc = r.get('energy_reduction_percent', 0)
                t_max_val = r.get('time_constraint', 0)
                # Get migration count from heuristic results dictionary key
                migr_count = r.get('migration_count', 'N/A')
                print(f"{r.get('config_name', 'N/A'):<30} | {r.get('initial_time', 0):>8.2f} | {r.get('final_time', 0):>8.2f} | {t_max_val:>8.2f} | "
                      f"{r.get('initial_energy', 0):>10.2f} | {r.get('final_energy', 0):>10.2f} | {valid_str:>5} | {migr_count:>4} | {reduc_perc:>6.1f}")
            else:
                 print(f"{'Malformed/Failed Result':<30} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>5} | {'N/A':>4} | {'N/A':>6}")
        print("-" * 98)

        # Deadline violation check remains the same
        violation_count = 0
        for r in all_results:
             if r and 'final_time' in r and 'time_constraint' in r and 'error' not in r:
                if r.get('final_time', 0) > r.get('time_constraint', float('inf')) + 1e-6:
                     print(f"  WARNING: Config '{r.get('config_name')}' - Final Time {r.get('final_time'):.2f} exceeds T_max {r.get('time_constraint'):.2f}")
                     violation_count += 1
        if violation_count == 0 and all_results:
            print("  All successful runs met their T_max constraint.")
        elif all_results:
            print(f"  {violation_count} successful run(s) potentially violated the T_max constraint.")

    if all_failures:
        print("\nFailures Encountered:")
        for f in all_failures:
            print(f" - Config: {f.get('config_name', 'Unknown Config')}")
            print(f"   Error: {f.get('error', 'Unknown Error').splitlines()[0]}")

    print("\n--- End of Overall Heuristic Test Run ---")
