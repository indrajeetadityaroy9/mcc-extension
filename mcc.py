#mcc.py
from copy import deepcopy
import bisect
from collections import deque
import numpy as np
from heapq import heappush, heappop
import os
import time
import pandas as pd

# Import necessary functions from data.py
from data import (
    ExecutionTier, SchedulingState, TaskMigrationState, core_execution_times, cloud_execution_times,
    add_task_attributes, generate_configs, MCCConfiguration
)
from validation import validate_task_dependencies_no_edge


class Task(object):
    def __init__(self, id, pred_tasks=None, succ_task=None, core_times=None, cloud_times=None,
                 edge_times=None, # Added edge_times
                 num_cores=3, num_edge_nodes=0, num_edge_cores=0, # Added edge params
                 complexity=None, data_intensity=None):
        """
        Initialize a task node in the directed acyclic graph.
        
        Args:
            id: Task identifier (vi in the paper)
            pred_tasks: List of immediate predecessor tasks
            succ_task: List of immediate successor tasks
            core_times: Optional dictionary mapping task IDs to core execution times
            cloud_times: Optional dictionary mapping task IDs to cloud execution times
            num_cores: Number of cores available in the device (defaults to 3)
        """
        # Basic task graph structure
        self.id = id
        self.pred_tasks = pred_tasks or []
        self.succ_tasks = succ_task or []

        # Use provided execution times or fall back to globals
        if core_times is not None:
            self.local_execution_times = core_times.get(id, [])
        else:
            self.local_execution_times = core_execution_times.get(id, [])
            
        if cloud_times is not None:
            self.cloud_execution_times = cloud_times.get(id, [])
        else:
            self.cloud_execution_times = cloud_execution_times.get(id, [])

        # Task completion timing parameters
        self.FT_l = 0
        self.FT_ws = 0
        self.FT_c = 0
        self.FT_wr = 0

        # Ready Times
        self.RT_l = -1
        self.RT_ws = -1
        self.RT_c = -1
        self.RT_wr = -1

        # Task scheduling parameters
        self.priority_score = None

        # Execution assignment
        self.assignment = -2

        # Flag for local core vs cloud
        self.is_core_task = False

        # Dynamic initialization based on actual number of cores
        # Initialize with -1 for each core plus one for cloud
        self.execution_unit_task_start_times = [-1] * (num_cores + 1)

        # Final completion time for the task
        self.execution_finish_time = -1

        # Current state in scheduling algorithm
        self.is_scheduled = SchedulingState.UNSCHEDULED

        # Additional attributes
        self.complexity = None
        self.data_intensity = None
        self.data_sizes = {}
        self.task_type = None
        self.execution_tier = None
        self.device_core = -1
    
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
        # Default fallback if tier/times are inconsistent
        return max(self.FT_l, self.FT_wr,
                   max(self.FT_edge_receive.values()) if self.FT_edge_receive else 0.0,
                   0.0) # Ensure non-negative


def total_time(tasks):
    # Implementation of total completion time calculation T_total from equation (10):
    # T_total = max(max(FTi^l, FTi^wr))
    #           vi∈exit_tasks

    # Find the maximum completion time among all exit tasks
    return max(
        # For each exit task vi, compute max(FTi^l, FTi^wr) where:
        # - FTi^l: Finish time if task executes on local core
        # - FTi^wr: Finish time if task executes on cloud (when results are received)
        # If FTi^l = 0, task executed on cloud
        # If FTi^wr = 0, task executed locally
        max(task.FT_l, task.FT_wr)

        # Only consider exit tasks (tasks with no successors)
        # This implements the "exit tasks" condition from equation (10)
        for task in tasks
        if not task.succ_tasks
    )


def total_energy(tasks, device_power_profiles, rf_power, upload_rates):
    total = 0.0

    for task in tasks:
        if task.is_core_task:
            # For local tasks
            core_id = task.assignment
            if core_id < 0 or core_id >= len(device_power_profiles):
                continue  # Skip invalid assignments

            core_info = device_power_profiles.get(core_id, {})
            idle_pwr = core_info.get('idle_power', 0.1)

            # Get dynamic power function or default
            if 'dynamic_power' in core_info and callable(core_info['dynamic_power']):
                dyn_func = core_info['dynamic_power']
            else:
                dyn_func = lambda load: 1.0 * load

            # Assume full load while task is running
            pwr_during_task = idle_pwr + dyn_func(1.0)

            # Get execution time
            if core_id < len(task.local_execution_times):
                exec_time = task.local_execution_times[core_id]
                task_energy = pwr_during_task * exec_time
                total += task_energy
        else:
            # For cloud tasks
            data_rate_mbps = upload_rates.get('device_to_cloud', 5.0)

            # Get RF power function or default
            if 'device_to_cloud' in rf_power and callable(rf_power['device_to_cloud']):
                rf_func = rf_power['device_to_cloud']
                radio_pwr = rf_func(data_rate_mbps, 70.0)  # Default signal strength
            else:
                radio_pwr = 1.0  # Default power

            # Calculate energy for sending task to cloud
            send_time = task.cloud_execution_times[0]
            task_energy = radio_pwr * send_time
            total += task_energy

    return total


def primary_assignment(tasks):
    """
    Implements the "Primary Assignment" phase described in Section III.A.1.
    This is the first phase of the initial scheduling algorithm that determines
    which tasks should be considered for cloud execution.
    """
    for task in tasks:
        # Calculate T_i^l_min (minimum local execution time)
        # Implements equation (11): T_i^l_min = min(1≤k≤K) T_i,k^l
        # where K is the number of local cores
        # This represents the best-case local execution scenario
        # by choosing the fastest available core
        t_l_min = min(task.local_execution_times) if task.local_execution_times else float('inf')

        # Calculate T_i^re (remote execution time)
        # Implements equation (12): T_i^re = T_i^s + T_i^c + T_i^r
        # This represents total time for cloud execution including:
        # T_i^s: Time to send task specification and input data
        # T_i^c: Time for cloud computation
        # T_i^r: Time to receive results
        t_re = (task.cloud_execution_times[0] +  # T_i^s (send)
                task.cloud_execution_times[1] +  # T_i^c (cloud)
                task.cloud_execution_times[2])  # T_i^r (receive)

        # Task assignment decision
        # If T_i^re < T_i^l_min, offloading to cloud will save time
        # This implements the cloud task selection criteria from Section III.A.1:
        if t_re < t_l_min:
            task.is_core_task = False  # Mark as cloud task
            task.execution_tier = ExecutionTier.CLOUD
        else:
            task.is_core_task = True  # Mark for local execution
            task.execution_tier = ExecutionTier.DEVICE


def task_prioritizing(tasks):
    """
    Implements the "Task Prioritizing" phase described in Section III.A.2.
    Calculates priority levels for each task to determine scheduling order.
    """
    w = [0] * len(tasks)
    # Step 1: Calculate computation costs (wi) for each task
    for i, task in enumerate(tasks):
        if not task.is_core_task:
            # For cloud tasks:
            # Implement equation (13): wi = Ti^re
            # where Ti^re is total remote execution time including:
            # - Data sending time (Ti^s)
            # - Cloud computation time (Ti^c)
            # - Result receiving time (Ti^r)
            w[i] = (task.cloud_execution_times[0] +  # Ti^s
                    task.cloud_execution_times[1] +  # Ti^c
                    task.cloud_execution_times[2])  # Ti^r
        else:
            # For local tasks:
            # Implement equation (14): wi = avg(1≤k≤K) Ti,k^l
            # Average computation time across all K cores
            w[i] = sum(task.local_execution_times) / len(
                task.local_execution_times) if task.local_execution_times else 0

    # Cache for memoization of priority calculations
    # This optimizes recursive calculations for tasks in cycles
    computed_priority_scores = {}

    def calculate_priority(task):
        """
        Recursive implementation of priority calculation.
        Implements equations (15) and (16) from the paper.
        """
        # Memoization check
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        # Base case: Exit tasks
        # Implement equation (16): priority(vi) = wi for vi ∈ exit_tasks
        if not task.succ_tasks:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]

        # Recursive case: Non-exit tasks
        # Implement equation (15):
        # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        # This represents length of critical path from task to exit
        max_successor_priority = max(calculate_priority(successor)
                                     for successor in task.succ_tasks)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks using recursive algorithm
    for task in tasks:
        calculate_priority(task)

    # Update priority scores in task objects
    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]


class InitialTaskScheduler:
    """
    Implements the initial scheduling algorithm described in Section III.A.
    This is Step One of the two-step algorithm, focusing on minimal-delay scheduling.
    """

    def __init__(self, tasks, num_cores=3):
        """
        Initialize scheduler with tasks and resources.
        Tracks timing for both local cores and cloud communication channels.
        """
        self.tasks = tasks
        self.k = num_cores  # K cores from paper

        # Resource timing tracking (Section II.B and II.C)
        self.core_earliest_ready = [0] * self.k  # When each core becomes available
        self.ws_ready = 0  # Next available time for RF sending channel
        self.wr_ready = 0  # Next available time for RF receiving channel

        # Sk sequence sets from Section III.B
        # Tracks task execution sequences for each resource (cores + cloud)
        self.sequences = [[] for _ in range(self.k + 1)]

    def get_priority_ordered_tasks(self):
        """
        Orders tasks by priority scores calculated in task_prioritizing().
        Implements ordering described in Section III.A.2, equation (15).
        """
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)  # Higher priority first
        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Separates tasks into entry and non-entry tasks while maintaining priority order.
        Implements task classification from Section II.A and III.A.3 of the paper:
        - Entry tasks: Starting points in the task graph with no dependencies
        - Non-entry tasks: Tasks that must wait for predecessors to complete
        """
        entry_tasks = []
        non_entry_tasks = []

        # Process tasks in priority order (from equation 15)
        # This ensures high-priority tasks are scheduled first
        for task_id in priority_order:
            task = self.tasks[task_id - 1]

            # Check if task has predecessors (pred(vi) from paper)
            if not task.pred_tasks:
                # Entry tasks have no predecessors and can start immediately
                # These correspond to v1 in Figure 1 of the paper
                entry_tasks.append(task)
            else:
                # Non-entry tasks must wait for predecessors to complete
                # Their ready times (RT) will be calculated based on predecessor finish times
                non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def identify_optimal_local_core(self, task, ready_time=0):
        """
        Finds optimal local core assignment for a task to minimize finish time.
        Implements the local core selection logic from Section III.A.3 for tasks
        that will be executed locally rather than offloaded to the cloud.
        """
        # Initialize with worst-case values
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        # Try each available core k (1 ≤ k ≤ K)
        for core in range(self.k):
            # Calculate earliest possible start time on this core
            # Must be after both:
            # 1. Task's ready time RTi^l (based on predecessors)
            # 2. Core's earliest available time (when previous task finishes)
            start_time = max(ready_time, self.core_earliest_ready[core])

            # Ensure the core index is valid for this task
            if core < len(task.local_execution_times):
                # Calculate finish time FTi^l using:
                # - Start time determined above
                # - Task's execution time on this core (Ti,k^l)
                finish_time = start_time + task.local_execution_times[core]

                # Keep track of core that gives earliest finish time
                # This implements the "minimizes the task's finish time"
                # criteria from Section III.A.3
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_core = core
                    best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Assigns a task to a local core and updates all relevant timing information.
        Implements local task scheduling from Section II.C.1 of the paper.
        """
        # Set task finish time on local core (FTi^l)
        # This is used in equation (10) for total completion time
        task.FT_l = finish_time
        # Set overall execution finish time
        # Used for precedence constraints and scheduling subsequent tasks
        task.execution_finish_time = finish_time
        # Initialize execution start times array
        # Index 0 to k: local cores
        # Index k+1: cloud
        # -1 indicates not scheduled on that unit
        task.execution_unit_task_start_times = [-1] * (self.k + 1)
        # Record actual start time on assigned core
        # This maintains the scheduling sequence Sk from Section III.B
        task.execution_unit_task_start_times[core] = start_time
        # Update core availability for next task
        # Core k cannot execute another task until current task finishes
        self.core_earliest_ready[core] = finish_time
        # Set task assignment (ki from Section II.B)
        # ki > 0 indicates local core execution
        task.assignment = core
        task.device_core = core
        # Mark task as scheduled in initial scheduling phase
        task.is_scheduled = SchedulingState.SCHEDULED
        # Add task to execution sequence for this core
        # This implements Sk sequence tracking from Section III.B
        # Used later for task migration phase
        self.sequences[core].append(task.id)

    def calculate_cloud_phases_timing(self, task):
        """
        Calculates timing for the three-phase cloud execution model described in Section II.B.
        Implements timing calculations from equations (1), (2), (4), (5), and (6).
        """
        # Phase 1: RF Sending Phase
        # Ready time RTi^ws from equation (4) - when we can start sending
        send_ready = task.RT_ws
        # Finish time FTi^ws = RTi^ws + Ti^s
        # Ti^s = datai/R^s from equation (1)
        # Time to send task specification and input data
        send_finish = send_ready + task.cloud_execution_times[0]
        # Phase 2: Cloud Computing Phase
        # Ready time RTi^c from equation (5)
        # Can start computing once sending is complete
        cloud_ready = send_finish
        # Finish time FTi^c = RTi^c + Ti^c
        # Ti^c is cloud computation time
        cloud_finish = cloud_ready + task.cloud_execution_times[1]
        # Phase 3: RF Receiving Phase
        # Ready time RTi^wr from equation (6)
        # Can start receiving once cloud computation finishes
        receive_ready = cloud_finish
        # Finish time FTi^wr considering:
        # 1. When results are ready (receive_ready)
        # 2. When wireless channel is available (wr_ready)
        # 3. Time to receive results Ti^r = data'i/R^r from equation (2)
        receive_finish = (
                max(self.wr_ready, receive_ready) +
                task.cloud_execution_times[2]
        )

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Schedules a task for cloud execution, updating all timing parameters and resource availability.
        Implements cloud scheduling from Section II.B and II.C.2 of the paper.
        """
        # Set timing parameters for three-phase cloud execution
        # Phase 1: RF Sending Phase
        task.RT_ws = send_ready  # When we can start sending (eq. 4)
        task.FT_ws = send_finish  # When sending completes (eq. 1)

        # Phase 2: Cloud Computing Phase
        task.RT_c = cloud_ready  # When cloud can start (eq. 5)
        task.FT_c = cloud_finish  # When cloud computation ends

        # Phase 3: RF Receiving Phase
        task.RT_wr = receive_ready  # When results are ready (eq. 6)
        task.FT_wr = receive_finish  # When results are received

        # Set overall execution finish time for precedence checking
        task.execution_finish_time = receive_finish

        # Clear local core finish time since executing on cloud
        # FTi^l = 0 indicates cloud execution as per Section II.C
        task.FT_l = 0

        # Initialize execution unit timing array
        # -1 indicates not scheduled on that unit
        task.execution_unit_task_start_times = [-1] * (self.k + 1)

        # Record cloud execution start time
        # Used for Sk sequence tracking in Section III.B
        task.execution_unit_task_start_times[self.k] = send_ready

        # Set task assignment (ki = 0 for cloud execution)
        # As specified in Section II.B
        task.assignment = self.k

        # Mark task as scheduled in initial phase
        task.is_scheduled = SchedulingState.SCHEDULED

        # Update wireless channel availability
        # Cannot send new task until current send completes
        self.ws_ready = send_finish
        # Cannot receive new results until current receive completes
        self.wr_ready = receive_finish

        # Add to cloud execution sequence
        # Maintains Sk sequences for task migration phase
        self.sequences[self.k].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """
        Schedules tasks with no predecessors (pred(vi) = ∅).
        Implements initial task scheduling from Section III.A.3 of the paper.
        Handles both local and cloud execution paths with appropriate ordering.
        """
        # Track tasks marked for cloud execution
        # These are scheduled after local tasks to enable pipeline staggering
        cloud_entry_tasks = []

        # First Phase: Schedule tasks assigned to local cores
        # Process local tasks first since they don't have sending dependencies
        for task in entry_tasks:
            if task.is_core_task:
                # Find optimal core assignment using criteria from Section III.A.3
                # Returns core k that minimizes finish time FTi^l
                core, start_time, finish_time = self.identify_optimal_local_core(task)

                # Schedule on chosen core
                # Updates task timing and core availability
                self.schedule_on_local_core(task, core, start_time, finish_time)
            else:
                # Collect cloud tasks for second phase
                cloud_entry_tasks.append(task)

        # Second Phase: Schedule cloud tasks
        # Process after local tasks to manage wireless channel congestion
        for task in cloud_entry_tasks:
            # Set wireless send ready time RTi^ws
            # Uses current wireless channel availability
            task.RT_ws = self.ws_ready

            # Calculate timing for three-phase cloud execution
            # Returns timing parameters for send, compute, and receive phases
            timing = self.calculate_cloud_phases_timing(task)

            # Schedule cloud execution
            # Updates task timing and wireless channel availability
            self.schedule_on_cloud(task, *timing)

    def calculate_non_entry_task_ready_times(self, task):
        """
        Calculates ready times for tasks that have predecessors.
        Implements equations (3) and (4) from Section II.C of the paper.
        """
        # Calculate local core ready time RTi^l (equation 3)
        # RTi^l = max(vj∈pred(vi)) max(FTj^l, FTj^wr)
        # Task can start on local core when all predecessors are complete:
        # - FTj^l: If predecessor executed locally
        # - FTj^wr: If predecessor executed on cloud
        task.RT_l = max(
            max(max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks),
            0  # Ensure non-negative ready time
        )

        # Calculate cloud sending ready time RTi^ws (equation 4)
        # RTi^ws = max(vj∈pred(vi)) max(FTj^l, FTj^ws)
        # Can start sending to cloud when:
        # 1. All predecessors have completed:
        #    - FTj^l: If predecessor executed locally
        #    - FTj^ws: If predecessor was sent to cloud
        # 2. Wireless sending channel is available
        task.RT_ws = max(
            max(max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks),
            self.ws_ready  # Channel availability
        )

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks that have predecessors (pred(vi) ≠ ∅).
        Implements execution unit selection from Section III.A.3 of the paper.

        Makes scheduling decisions by:
        1. Calculating ready times based on predecessors
        2. Evaluating both local and cloud execution options
        3. Selecting execution path that minimizes finish time
        """
        # Process tasks in priority order (from task_prioritizing)
        for task in non_entry_tasks:
            # Calculate RTi^l and RTi^ws based on predecessor finish times
            # Implements equations (3) and (4)
            self.calculate_non_entry_task_ready_times(task)

            # If task was marked for cloud in primary assignment
            if not task.is_core_task:
                # Calculate three-phase cloud execution timing
                timing = self.calculate_cloud_phases_timing(task)
                # Schedule task on cloud
                self.schedule_on_cloud(task, *timing)
            else:
                # For tasks marked for local execution:
                # 1. Find best local core option
                core, start_time, finish_time = self.identify_optimal_local_core(
                    task, task.RT_l  # Consider ready time RTi^l
                )

                # 2. Calculate cloud execution option for comparison
                # "schedule task vi on the core or offload it to the cloud
                # such that the finish time is minimized"
                timing = self.calculate_cloud_phases_timing(task)
                cloud_finish_time = timing[-1]  # FTi^wr

                # 3. Choose execution path with earlier finish time
                # This implements the minimum finish time criteria
                # from Section III.A.3
                if finish_time <= cloud_finish_time:
                    # Local execution is faster
                    self.schedule_on_local_core(task, core, start_time, finish_time)
                else:
                    # Cloud execution is faster
                    # Override primary assignment decision
                    task.is_core_task = False
                    task.execution_tier = ExecutionTier.CLOUD
                    self.schedule_on_cloud(task, *timing)

def execution_unit_selection(tasks, num_cores=3):
    """
    Implements execution unit selection phase described in Section III.A.3.
    This is the third and final phase of the initial scheduling algorithm
    after primary assignment and task prioritizing.

    Args:
        tasks: List of tasks from the application graph G=(V,E)
        num_cores: Number of cores in the mobile device (defaults to 3 for backward compatibility)

    Returns:
        sequences: List of task sequences Sk for each execution unit
                  Used in task migration phase for energy optimization
    """
    # 1. Primary Assignment (Initial Guess)
    primary_assignment(tasks)

    # 2. Task Prioritizing (Based on Critical Path)
    task_prioritizing(tasks)

    # Initialize scheduler with tasks and the specified number of cores
    scheduler = InitialTaskScheduler(tasks, num_cores)

    # Order tasks by priority score from equation (15)
    # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
    # Higher priority indicates task is on critical path
    priority_orderered_tasks = scheduler.get_priority_ordered_tasks()

    # Classify tasks based on dependencies
    # Entry tasks: pred(vi) = ∅ (can start immediately)
    # Non-entry tasks: must wait for predecessors
    # Maintains priority ordering within each category
    entry_tasks, non_entry_tasks = scheduler.classify_entry_tasks(priority_orderered_tasks)

    # Two-phase scheduling process:
    # 1. Schedule entry tasks (no dependencies)
    #    - Process local tasks first
    #    - Then handle cloud tasks with pipeline staggering
    scheduler.schedule_entry_tasks(entry_tasks)

    # 2. Schedule non-entry tasks (with dependencies)
    #    - Calculate ready times based on predecessors
    #    - Compare local vs cloud execution times
    #    - Choose path that minimizes finish time
    scheduler.schedule_non_entry_tasks(non_entry_tasks)

    # Update execution_tier for all tasks to maintain compatibility with utils.validate_task_dependencies
    for task in tasks:
        if task.is_core_task:
            task.execution_tier = ExecutionTier.DEVICE
            task.device_core = task.assignment
        else:
            task.execution_tier = ExecutionTier.CLOUD
            task.device_core = -1

    # Return task sequences for each execution unit
    # These Sk sequences are used in Section III.B
    # for the task migration algorithm
    return scheduler.sequences

def construct_sequence(tasks, task_id, execution_unit, original_sequence):
    """
   Implements the linear-time rescheduling algorithm described in Section III.B.2.
   Constructs new sequence after task migration while preserving task precedence.

   Args:
       tasks: List of all tasks in the application
       task_id: ID of task v_tar being migrated
       execution_unit: New execution location k_tar
       original_sequence: Current Sk sequences for all execution units

   Returns:
       Modified sequence sets after migrating the task
   """
    # Step 1: Create task lookup dictionary for O(1) access
    task_id_to_task = {task.id: task for task in tasks}

    # Step 2: Get the target task v_tar for migration
    target_task = task_id_to_task.get(task_id)
    
    # Get number of cores from the sequence length 
    num_cores = len(original_sequence) - 1
    
    # Get cloud index (equals the number of cores)
    cloud_index = num_cores

    # Step 3: Get ready time for insertion
    # RTi^l for local cores (k_tar < cloud_index)
    # RTi^ws for cloud execution (k_tar = cloud_index)
    target_task_rt = target_task.RT_l if target_task.is_core_task else target_task.RT_ws

    # Step 4: Remove task from original sequence
    # Implementation of equation (17):
    # "we will not change the ordering of tasks in the other cores"
    original_assignment = target_task.assignment
    original_sequence[original_assignment].remove(target_task.id)

    # Step 5: Get sequence for new execution unit
    # Prepare for ordered insertion based on start times
    new_sequence_task_list = original_sequence[execution_unit]

    # Get start times for tasks in new sequence
    # Used to maintain proper task ordering
    start_times = [
        task_id_to_task[task_id].execution_unit_task_start_times[execution_unit]
        for task_id in new_sequence_task_list
    ]

    # Step 6: Find insertion point using binary search
    # Implements "insert v_tar into S_k_tar such that v_tar is
    # executed after all its transitive predecessors and before
    # all its transitive successors"
    insertion_index = bisect.bisect_left(start_times, target_task_rt)

    # Step 7: Insert task at correct position
    # Maintains ordered sequence based on start times
    new_sequence_task_list.insert(insertion_index, target_task.id)

    # Step 8: Update task execution information
    # Set new assignment k_i and execution type
    target_task.assignment = execution_unit
    
    # Update is_core_task based on whether execution_unit is cloud (not hardcoded 3)
    target_task.is_core_task = (execution_unit != cloud_index)

    # Update execution tier and device core
    if target_task.is_core_task:
        target_task.execution_tier = ExecutionTier.DEVICE
        target_task.device_core = execution_unit
    else:
        target_task.execution_tier = ExecutionTier.CLOUD
        target_task.device_core = -1

    return original_sequence


class KernelScheduler:
    """
    Implements the kernel (rescheduling) algorithm from Section III.B.2.
    Provides linear-time rescheduling for task migration phase of MCC scheduling.

    This scheduler maintains state for:
    - Task dependency tracking
    - Resource availability
    - Task sequencing
    """

    def __init__(self, tasks, sequences):
        """
        Initialize kernel scheduler for task migration rescheduling.
        
        Args:
            tasks: List of Task objects to schedule
            sequences: Initial task sequences for each execution unit
        """
        self.tasks = tasks
        # Sk sequences from equation (17)
        # sequences[k]: Tasks assigned to execution unit k
        # k = 0,1,...,(num_cores-1): Local cores
        # k = num_cores: Cloud execution
        self.sequences = sequences
        
        # Determine the number of cores from the length of sequences
        # sequences includes entries for all cores plus one for cloud
        self.num_cores = len(sequences) - 1 if sequences else 3

        # Resource timing trackers - dynamically sized for the actual number of cores
        # Track when each execution unit becomes available

        # RTi^l ready times for local cores (k > 0)
        # From equation (3): When each core can start next task
        self.RT_ls = [0] * self.num_cores

        # Ready times for cloud execution phases
        # [0]: RTi^ws - Wireless sending (eq. 4)
        # [1]: RTi^c  - Cloud computation (eq. 5)
        # [2]: RTi^wr - Result receiving (eq. 6)
        self.cloud_phases_ready_times = [0] * 3

        # Initialize task readiness tracking vectors
        # These implement the ready1 and ready2 vectors
        # described in Section III.B.2
        self.dependency_ready, self.sequence_ready = self.initialize_task_state()

    def initialize_task_state(self):
        """
        Initializes task readiness tracking vectors described in Section III.B.2.
        These vectors enable linear-time rescheduling by tracking:
        1. Task dependency completion (ready1)
        2. Sequence position readiness (ready2)
        """
        # Initialize ready1 vector (dependency tracking)
        # ready1[j] is number of immediate predecessors not yet scheduled
        # "ready1[j] is the number of immediate predecessors of task v[j]
        # that have not been scheduled"
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]

        # Initialize ready2 vector (sequence position tracking)
        # ready2[j] indicates if task is ready in its sequence:
        # -1: Task not in current sequence
        #  0: Task ready to execute (first in sequence or predecessor completed)
        #  1: Task waiting for predecessor in sequence
        sequence_ready = [-1] * len(self.tasks)

        # Process each execution sequence Sk
        for sequence in self.sequences:
            if sequence:  # Non-empty sequence
                # Mark first task in sequence as ready
                # "ready2[j] = 0 if all the tasks before task v[j] in
                # the same sequence have already been scheduled"
                sequence_ready[sequence[0] - 1] = 0

        return dependency_ready, sequence_ready

    def update_task_state(self, task):
        """
        Updates readiness vectors (ready1, ready2) for a task after scheduling changes.
        Implements the vector update logic from Section III.B.2 of the paper.

        This maintains the invariants required for linear-time scheduling:
        1. ready1[j] tracks unscheduled predecessors
        2. ready2[j] tracks sequence position readiness
        """
        # Only update state for unscheduled tasks
        # Once a task is KERNEL_SCHEDULED, its state is final
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update ready1 vector (dependency tracking)
            # "ready1[j] by one for all vj ∈ succ(vi)"
            # Count immediate predecessors that haven't been scheduled
            self.dependency_ready[task.id - 1] = sum(
                1 for pred_task in task.pred_tasks
                if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )

            # Update ready2 vector (sequence position)
            # Find task's position in its current execution sequence
            for sequence in self.sequences:
                if task.id in sequence:
                    idx = sequence.index(task.id)
                    if idx > 0:
                        # Task has predecessor in sequence
                        # Check if predecessor has been scheduled
                        prev_task = self.tasks[sequence[idx - 1] - 1]
                        self.sequence_ready[task.id - 1] = (
                            # 1: Waiting for predecessor
                            # 0: Predecessor completed
                            1 if prev_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
                            else 0
                        )
                    else:
                        # First task in sequence
                        # "ready2[j] = 0 if all the tasks before task vj
                        # in the same sequence have already been scheduled"
                        self.sequence_ready[task.id - 1] = 0
                    break

    def schedule_local_task(self, task):
        """
        Schedules a task for local core execution following Section II.C.1.
        Implements timing calculations for tasks executing on cores.

        Updates:
            - RTi^l: Local execution ready time (eq. 3)
            - FTi^l: Local execution finish time
            - Core availability tracking
        """
        # Calculate ready time RTi^l for local execution
        # Implements equation (3): RTi^l = max(vj∈pred(vi)) max(FTj^l, FTj^wr)
        if not task.pred_tasks:
            # Entry tasks can start immediately
            task.RT_l = 0
        else:
            # Find latest completion time among predecessors
            # Consider both local (FTj^l) and cloud (FTj^wr) execution
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks
            )
            task.RT_l = max(pred_task_completion_times, default=0)

        # Schedule on assigned core k
        core_index = task.assignment
        # Initialize execution timing array
        # Index 0-2: Local cores
        # Index 3: Cloud
        task.execution_unit_task_start_times = [-1] * (self.num_cores + 1)

        # Calculate actual start time considering:
        # 1. Task ready time RTi^l
        # 2. Core availability (RT_ls[k])
        task.execution_unit_task_start_times[core_index] = max(
            self.RT_ls[core_index],  # Core availability
            task.RT_l  # Task ready time
        )

        # Calculate finish time FTi^l
        # FTi^l = start_time + Ti,k^l
        # where Ti,k^l is execution time on core k
        task.FT_l = (
                task.execution_unit_task_start_times[core_index] +
                task.local_execution_times[core_index]
        )

        # Update core k's next available time
        self.RT_ls[core_index] = task.FT_l

        # Clear cloud execution timings
        # FTi^ws = FTi^c = FTi^wr = 0 for local tasks
        # As specified in Section II.C
        task.FT_ws = -1
        task.FT_c = -1
        task.FT_wr = -1

    def schedule_cloud_task(self, task):
        """
        Schedules three-phase cloud execution described in Section II.B.
        Implements timing calculations for:
        1. RF sending phase (equations 1, 4)
        2. Cloud computation phase (equation 5)
        3. RF receiving phase (equations 2, 6)

        Updates all cloud execution timing parameters and resource availability
        """
        # Calculate wireless sending ready time RTi^ws
        # Implements equation (4): RTi^ws = max(vj∈pred(vi)) max(FTj^l, FTj^ws)
        if not task.pred_tasks:
            # Entry tasks can start sending immediately
            task.RT_ws = 0
        else:
            # Find latest completion time among predecessors
            # Consider both local execution (FTj^l) and cloud sending (FTj^ws)
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks
            )
            task.RT_ws = max(pred_task_completion_times)

        # Initialize timing array for execution units
        task.execution_unit_task_start_times = [-1] * (self.num_cores + 1)
        # Set cloud start time considering:
        # 1. Wireless channel availability
        # 2. Task ready time RTi^ws
        cloud_index = self.num_cores
        task.execution_unit_task_start_times[cloud_index] = max(
            self.cloud_phases_ready_times[0],  # Channel availability
            task.RT_ws  # Task ready time
        )

        # Phase 1: RF Sending Phase
        # Implement equation (1): Ti^s = datai/R^s
        # Calculate finish time FTi^ws
        cloud_index = self.num_cores
        task.FT_ws = (
                task.execution_unit_task_start_times[cloud_index] +
                task.cloud_execution_times[0]  # Ti^s
        )
        # Update sending channel availability
        self.cloud_phases_ready_times[0] = task.FT_ws

        # Phase 2: Cloud Computing Phase
        # Implement equation (5): RTi^c calculation
        task.RT_c = max(
            task.FT_ws,  # Must finish sending
            max((pred_task.FT_c for pred_task in task.pred_tasks), default=0)
        )
        # Calculate cloud finish time FTi^c
        task.FT_c = (
                max(self.cloud_phases_ready_times[1], task.RT_c) +
                task.cloud_execution_times[1]  # Ti^c
        )
        # Update cloud availability
        self.cloud_phases_ready_times[1] = task.FT_c

        # Phase 3: RF Receiving Phase
        # Implement equation (6): RTi^wr = FTi^c
        task.RT_wr = task.FT_c
        # Calculate receiving finish time using equation (2)
        task.FT_wr = (
                max(self.cloud_phases_ready_times[2], task.RT_wr) +
                task.cloud_execution_times[2]  # Ti^r
        )
        # Update receiving channel availability
        self.cloud_phases_ready_times[2] = task.FT_wr

        # Clear local execution timing
        # FTi^l = 0 for cloud tasks as per Section II.C
        task.FT_l = -1

    def initialize_queue(self):
        """
        Initializes LIFO stack for linear-time scheduling described in Section III.B.2.
        Identifies initially ready tasks based on both dependency and sequence readiness.
        """
        # Create LIFO stack (implemented as deque)
        # A task vi is ready for scheduling when both:
        # 1. ready1[i] = 0: All predecessors scheduled
        # 2. ready2[i] = 0: Ready in execution sequence
        return deque(
            task for task in self.tasks
            if (
                # Check sequence readiness (ready2[i] = 0)
                # Task must be first in sequence or after scheduled task
                    self.sequence_ready[task.id - 1] == 0
                    and
                    # Check dependency readiness (ready1[i] = 0)
                    # All predecessors must be completely scheduled
                    all(pred_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED
                        for pred_task in task.pred_tasks)
            )
        )


def kernel_algorithm(tasks, sequences):
    """
   Implements the kernel (rescheduling) algorithm from Section III.B.2.
   Provides linear-time task rescheduling for the task migration phase.
   
   Args:
       tasks: List of Task objects to be scheduled
       sequences: Task sequences for each execution unit
       
   Returns:
       Updated tasks with new scheduling information
   """
    # Initialize kernel scheduler with tasks and sequences
    # Note: number of cores is automatically derived from sequences
    scheduler = KernelScheduler(tasks, sequences)

    # Initialize LIFO stack with ready tasks
    # "initialized by pushing the task vi's with both
    # ready1[i] = 0 and ready2[i] = 0 into the empty stack"
    queue = scheduler.initialize_queue()

    # Main scheduling loop
    # "repeat the following steps until the stack becomes empty"
    while queue:
        # Pop next ready task from stack
        current_task = queue.popleft()
        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution type
        if current_task.is_core_task:
            # Schedule on assigned local core k
            # Updates RTi^l and FTi^l
            scheduler.schedule_local_task(current_task)
        else:
            # Schedule three-phase cloud execution
            # Updates RTi^ws, FTi^ws, RTi^c, FTi^c, RTi^wr, FTi^wr
            scheduler.schedule_cloud_task(current_task)

        # Update ready1 and ready2 vectors
        # "Update vectors ready1 (reducing ready1[j] by one for all
        # vj ∈ succ(vi)) and ready2, and push all the new tasks vj
        # with both ready1[j] = 0 and ready2[j] = 0 into the stack"
        for task in tasks:
            scheduler.update_task_state(task)

            # Add newly ready tasks to stack
            if (scheduler.dependency_ready[task.id - 1] == 0 and  # ready1[j] = 0
                    scheduler.sequence_ready[task.id - 1] == 0 and  # ready2[j] = 0
                    task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
                    task not in queue):
                queue.append(task)

    # Reset scheduling state for next iteration
    # Allows multiple runs of kernel algorithm during task migration
    for task in tasks:
        task.is_scheduled = SchedulingState.UNSCHEDULED

    return tasks


def generate_cache_key(tasks, task_idx, target_execution_unit):
    """
        Generates cache key for memoizing migration evaluations.
        Enables efficient evaluation of migration options in Section III.B.
        """
    # Create cache key from:
    # 1. Task being migrated (v_tar)
    # 2. Target execution unit (k_tar)
    # 3. Current task assignments (ki for all tasks)
    return (task_idx, target_execution_unit,
            tuple(task.assignment for task in tasks))


def evaluate_migration(tasks, seqs, task_idx, target_execution_unit, migration_cache,
                       device_power_profiles, rf_power, upload_rates):
    """
    Evaluates potential task migration scenario using dynamic power models.

    Args:
        tasks: List of all tasks
        seqs: Current Sk sequences
        task_idx: Index of task v_tar to migrate
        target_execution_unit: Proposed location k_tar
        migration_cache: Dictionary storing previous migration evaluations
        device_power_profiles: Power models for device cores
        rf_power: RF component power models
        upload_rates: Data rates for different connections

    Returns:
        tuple: (T_total, E_total) after migration
    """
    # Generate cache key for this migration scenario
    cache_key = generate_cache_key(tasks, task_idx, target_execution_unit)

    # Check cache for previously evaluated scenario
    if cache_key in migration_cache:
        return migration_cache[cache_key]

    # Create copies to avoid modifying original state
    sequence_copy = [seq.copy() for seq in seqs]
    tasks_copy = deepcopy(tasks)

    # Apply migration and recalculate schedule
    sequence_copy = construct_sequence(
        tasks_copy,
        task_idx + 1,  # Convert to 1-based task ID
        target_execution_unit,
        sequence_copy
    )
    kernel_algorithm(tasks_copy, sequence_copy)

    # Calculate new metrics
    migration_T = total_time(tasks_copy)
    migration_E = total_energy(tasks_copy, device_power_profiles, rf_power, upload_rates)

    # Cache results
    migration_cache[cache_key] = (migration_T, migration_E)
    return migration_T, migration_E

def initialize_migration_choices(tasks, num_cores):
    """
    Initializes possible migration choices for each task as described in Section III.B.
    
    Args:
        tasks: List of all tasks
        num_cores: Number of cores in the system
        
    Returns:
        Migration possibility matrix
    """
    # Create matrix of migration possibilities:
    # N rows (tasks) x (num_cores + 1) columns (cores + cloud)
    # Implements "total of N × K migration choices"
    # from Section III.B outer loop
    migration_choices = np.zeros((len(tasks), num_cores + 1), dtype=bool)
    
    # Cloud index is equal to the number of cores
    cloud_index = num_cores

    # Set valid migration targets for each task
    for i, task in enumerate(tasks):
        if task.assignment == cloud_index:
            # Cloud-assigned tasks (ki = num_cores)
            # Can potentially migrate to any local core
            migration_choices[i, :] = True
        else:
            # Locally-assigned tasks (ki < num_cores)
            # Can only migrate to other cores or cloud
            # Maintains task's current valid execution options
            migration_choices[i, task.assignment] = True

    return migration_choices


def identify_optimal_migration(migration_trials_results, T_final, E_total, T_max):
    """
        Identifies optimal task migration as described in Section III.B.
        Implements two-step selection process for energy reduction while maintaining
        completion time constraints.
        """
    # Step 1: Find migrations that reduce energy without increasing time
    # "select the choice that results in the largest energy reduction
    # compared with the current schedule and no increase in T_total"
    best_energy_reduction = 0
    best_migration = None

    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Skip migrations violating T_max constraint
        if time > T_max:
            continue

        # Calculate potential energy reduction
        # ΔE = E_total_current - E_total_after
        energy_reduction = E_total - energy

        # Check if migration:
        # 1. Doesn't increase completion time (T_total)
        # 2. Reduces energy consumption (E_total)
        if time <= T_final and energy_reduction > 0:
            if energy_reduction > best_energy_reduction:
                best_energy_reduction = energy_reduction
                best_migration = (task_idx, resource_idx, time, energy)

    # Return best energy-reducing migration if found
    if best_migration:
        task_idx, resource_idx, time, energy = best_migration
        return TaskMigrationState(
            time=time,
            energy=energy,
            efficiency=best_energy_reduction,
            task_index=task_idx + 1,
            target_execution_unit=resource_idx + 1
        )

    # Step 2: If no direct energy reduction found
    # "select the one that results in the largest ratio of
    # energy reduction to the increase of T_total"
    migration_candidates = []
    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Maintain T_max constraint
        if time > T_max:
            continue

        # Calculate energy reduction
        energy_reduction = E_total - energy
        if energy_reduction > 0:
            # Calculate efficiency ratio
            # ΔE / ΔT where ΔT is increase in completion time
            time_increase = max(0, time - T_final)
            if time_increase == 0:
                efficiency = float('inf')  # Prioritize no time increase
            else:
                efficiency = energy_reduction / time_increase

            heappush(migration_candidates,
                     (-efficiency, task_idx, resource_idx, time, energy))

    if not migration_candidates:
        return None

    # Return migration with best efficiency ratio
    neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
    return TaskMigrationState(
        time=T_best,
        energy=E_best,
        efficiency=-neg_ratio,
        task_index=n_best + 1,
        target_execution_unit=k_best + 1
    )

def optimize_schedule(tasks, sequence, T_max, device_power_profiles, rf_power, upload_rates):
    """
    Implements the task migration algorithm from Section III.B of the paper.
    Optimizes energy consumption while maintaining completion time constraints.

    Args:
        tasks: List of tasks from application graph G=(V,E)
        sequence: Initial Sk sequences from minimal-delay scheduling
        T_final: Target completion time constraint T_max
        device_power_profiles: Power models for device cores
        rf_power: RF component power models
        upload_rates: Data rates for different connections

    Returns:
        tuple: (tasks, sequence) with optimized scheduling and list of migrations
    """
    # Get number of cores from sequence length
    num_cores = len(sequence) - 1
    
    # Cache for memoizing migration evaluations
    migration_cache = {}

    # Track migrations for reporting
    migrations = []

    # Calculate initial energy consumption E_total
    current_iteration_energy = total_energy(tasks, device_power_profiles, rf_power, upload_rates)

    # Iterative improvement loop
    energy_improved = True
    while energy_improved:
        # Store current energy for comparison
        previous_iteration_energy = current_iteration_energy

        # Get current schedule metrics
        current_time = total_time(tasks)  # T_total (equation 10)

        # Initialize migration possibilities matrix with correct core count
        migration_choices = initialize_migration_choices(tasks, num_cores)

        # Evaluate all valid migration options
        migration_trials_results = []
        for task_idx in range(len(tasks)):
            # Dynamic range based on actual core count
            for possible_execution_unit in range(num_cores + 1):
                if migration_choices[task_idx, possible_execution_unit]:
                    continue

                # Calculate T_total and E_total after migration
                migration_trial_time, migration_trial_energy = evaluate_migration(
                    tasks, sequence, task_idx, possible_execution_unit, migration_cache,
                    device_power_profiles, rf_power, upload_rates
                )
                migration_trials_results.append(
                    (task_idx, possible_execution_unit,
                     migration_trial_time, migration_trial_energy)
                )

        # Select best migration using two-step criteria
        # 1. Reduce energy without increasing time
        # 2. Best energy/time tradeoff ratio
        best_migration = identify_optimal_migration(
            migration_trials_results=migration_trials_results,
            T_final=current_time,
            E_total=previous_iteration_energy,
            T_max=T_max
        )

        # Exit if no valid migrations remain
        if best_migration is None:
            energy_improved = False
            break

        # Record the migration
        migrations.append({
            'task_id': best_migration.task_index,
            'from_unit': tasks[best_migration.task_index-1].assignment,
            'to_unit': best_migration.target_execution_unit - 1,
            'energy_before': previous_iteration_energy,
            'energy_after': best_migration.energy,
            'time_before': current_time,
            'time_after': best_migration.time,
            'efficiency': best_migration.efficiency
        })

        # Apply selected migration:
        # 1. Construct new sequences (Section III.B.2)
        sequence = construct_sequence(
            tasks,
            best_migration.task_index,
            best_migration.target_execution_unit - 1,
            sequence
        )

        # 2. Apply kernel algorithm for O(N) rescheduling
        kernel_algorithm(tasks, sequence)

        # Calculate new energy consumption
        current_iteration_energy = total_energy(tasks, device_power_profiles, rf_power, upload_rates)
        energy_improved = current_iteration_energy < previous_iteration_energy

        # Manage cache size for memory efficiency
        if len(migration_cache) > 1000:
            migration_cache.clear()

    return tasks, sequence, migrations

def create_task_graph(num_cores=3, core_times=None, cloud_times=None): # Added edge_times
    """
    Creates a predefined task graph structure (like Fig 1 in the paper).
    Passes execution times and system config to Task constructor.

    Args:
        num_tasks: Currently fixed at 20 for this structure.
        num_cores, num_edge_nodes, num_edge_cores: System config.
        core_times, cloud_times, edge_times: Dictionaries mapping task ID to execution times.

    Returns:
        List of Task objects with dependencies set.
    """

    # pred_tasks=None, succ_task=None, core_times=None, cloud_times=None, num_cores=3
    # Helper to create Task with all params
    def _create_task(task_id, succ_list=None):
        return Task(
            id=task_id,
            succ_task=succ_list or [], # Ensure it's a list
            core_times=core_times,
            cloud_times=cloud_times,
            num_cores=num_cores
        )

    # Create tasks (successors defined first for easier linking)
    task20 = _create_task(20)
    task19 = _create_task(19, [task20])
    task18 = _create_task(18, [task20])
    task17 = _create_task(17, [task20])
    task16 = _create_task(16, [task19])
    task15 = _create_task(15, [task19])
    task14 = _create_task(14, [task18, task19])
    task13 = _create_task(13, [task17, task18])
    task12 = _create_task(12, [task17])
    task11 = _create_task(11, [task15, task16])
    task10 = _create_task(10, [task11, task15])
    task9 = _create_task(9, [task13, task14])
    task8 = _create_task(8, [task12, task13])
    task7 = _create_task(7, [task12])
    task6 = _create_task(6, [task10, task11])
    task5 = _create_task(5, [task9, task10])
    task4 = _create_task(4, [task8, task9])
    task3 = _create_task(3, [task7, task8])
    task2 = _create_task(2, [task7])
    task1 = _create_task(1, [task7])

    # Set predecessor relationships programmatically
    all_tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,
                 task11, task12, task13, task14, task15, task16, task17, task18, task19, task20]
    task_map = {t.id: t for t in all_tasks}

    for task in all_tasks:
        task.pred_tasks = [] # Initialize empty list

    for task in all_tasks:
        for succ_task in task.succ_tasks:
             if succ_task.id in task_map:
                 task_map[succ_task.id].pred_tasks.append(task)
             else:
                  print(f"Error: Successor task ID {succ_task.id} not found in map for Task {task.id}")

    return all_tasks


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
    global core_execution_times, cloud_execution_times
    core_execution_times = params.get('core_execution_times', {})
    cloud_execution_times = params.get('cloud_execution_times', {})

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


def run_unified_test(config: MCCConfiguration):
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
    tasks = create_task_graph(
        num_cores=config.num_cores,
        core_times=params['core_execution_times'],
        cloud_times=params['cloud_execution_times']
    )

    # 3. Assign detailed task attributes
    tasks = assign_task_attributes(tasks, config)

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
        num_cores=config.num_cores
    )
    initial_schedule_time = time.time() - initial_start_time

    # *** Store the state of tasks *after* initial scheduling ***
    tasks_initial_state = deepcopy(tasks_for_initial_scheduling)

    # Calculate metrics for initial schedule (using the scheduled task state)
    T_initial = total_time(tasks_initial_state)
    E_initial = total_energy(tasks_initial_state, device_power_profiles, rf_power, upload_rates)
    print(f" Initial Schedule: T={T_initial:.2f}, E={E_initial:.2f} (took {initial_schedule_time:.2f}s)")

    initial_dist = {tier: 0 for tier in [ExecutionTier.DEVICE, ExecutionTier.CLOUD]} # 2-tier
    for task in tasks_initial_state:
         if task.execution_tier in initial_dist: initial_dist[task.execution_tier] += 1
         else: initial_dist[task.execution_tier] = 1 # Handle unexpected tiers
    print(f"  Initial Distribution: {initial_dist}")

    is_valid_initial, violations_initial = validate_task_dependencies_no_edge(tasks_initial_state)
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
        device_power_profiles, rf_power, upload_rates
    )
    optimize_schedule_time = time.time() - optimize_start_time

    # Calculate metrics for final schedule
    T_final = total_time(tasks_final)
    E_final = total_energy(tasks_final, device_power_profiles, rf_power, upload_rates)
    print(f" Optimized Schedule: T={T_final:.2f}, E={E_final:.2f} (took {optimize_schedule_time:.2f}s)")

    final_dist = {tier: 0 for tier in [ExecutionTier.DEVICE, ExecutionTier.CLOUD]} # 2-tier
    for task in tasks_final:
         if task.execution_tier in final_dist: final_dist[task.execution_tier] += 1
         else: final_dist[task.execution_tier] = 1 # Handle unexpected tiers
    print(f"  Final Distribution: {final_dist}")
    print(f"  Migrations: {len(migrations)}")

    is_valid_final, violations_final = validate_task_dependencies_no_edge(tasks_final)
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
        # *** Add the final sequence to the results ***
        'sequence_initial': sequence_initial,
        'sequence_final': sequence_final,
        'params': { # Keep relevant params
             'upload_rates': upload_rates,
             'download_rates': download_rates,
             'core_exec_times_sample': params['core_execution_times'].get(1),
             'cloud_exec_times_sample': params['cloud_execution_times'].get(1),
         }
    }
    result_data['deadline_met'] = (result_data['final_time'] <= result_data['time_constraint'])
    result_data['initial_local_count'] = initial_dist.get(ExecutionTier.DEVICE, 0)
    result_data['initial_cloud_count'] = initial_dist.get(ExecutionTier.CLOUD, 0)
    result_data['final_local_count'] = final_dist.get(ExecutionTier.DEVICE, 0)
    result_data['final_cloud_count'] = final_dist.get(ExecutionTier.CLOUD, 0)
    result_data['cloud_migration'] = result_data['final_cloud_count'] - result_data['initial_cloud_count']
    return result_data

def run_sequential_test_suite(param_ranges=None, sampling_method='grid', num_samples=1000, progress_interval=5, output_prefix="mcc_results"):
    """
    Runs a suite of MCC scheduling tests sequentially based on generated configurations.
    """
    # Validate parameters
    if sampling_method == 'random' and param_ranges is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive for random sampling, got {num_samples}")
    if sampling_method not in ['grid', 'random']:
         raise ValueError(f"sampling_method must be 'grid' or 'random', got {sampling_method}")

    # Generate test configurations
    print(f"Generating test configurations...")
    configs = generate_configs(
        param_ranges=param_ranges,
        sampling_method=sampling_method,
        num_samples=num_samples if sampling_method == 'random' else None
    )

    if not configs:
        print("Warning: No configurations were generated with the current settings. Exiting.")
        return pd.DataFrame(), [], None # Return empty results

    # Count specialized configs for reporting (ones with predefined core profiles)
    specialized_count = sum(1 for config in configs
                           if hasattr(config, 'core_power_profiles') and config.core_power_profiles)

    print(f"Generated {len(configs)} test configurations")
    if specialized_count > 0:
        print(f"Including {specialized_count} specialized test configurations")

    # Initialize result and failure lists
    results_with_config = [] # Store tuples of (config, result_dict)
    failures = []

    # Create timestamp for this test run
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    # Create a descriptive output directory
    method_str = f"{sampling_method}"
    if sampling_method == 'random':
        method_str += f"_{num_samples}"

    output_dir = f"{output_prefix}_{method_str}_{timestamp}"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        pass # Proceed, saving might fail

    # Save configuration details
    try:
        config_file_path = f'{output_dir}/test_configuration.txt'
        with open(config_file_path, 'w') as f:
             f.write(f"Test Configuration:\n")
             f.write(f"- Sampling method: {sampling_method}\n")
             if sampling_method == 'random':
                 f.write(f"- Number of samples: {num_samples}\n")
             if param_ranges:
                 f.write(f"- Parameter ranges:\n")
                 for param, val in param_ranges.items():
                      f.write(f"  - {param}: {val}\n")
             else:
                  f.write("- Parameter ranges: None\n")
             f.write(f"- Total configurations: {len(configs)}\n")
             f.write(f"- Specialized configurations count: {specialized_count}\n")
        print(f"Saved test configuration details to {config_file_path}")
    except IOError as e:
        print(f"Warning: Could not save test configuration details: {e}")

    print(f"\nExecuting {len(configs)} tests sequentially...")
    print(f"Results will be saved to: {output_dir}")

    success_count = 0

    for i, config in enumerate(configs):
        is_specialized = hasattr(config, 'core_power_profiles') and config.core_power_profiles
        config_type = "specialized" if is_specialized else "standard"
        test_start_time_local = time.time() # Renamed to avoid conflict

        try:
            # Run the test
            result_dict = run_unified_test(config) # Returns dict including 'config' object

            # *** Store the original config along with its result dictionary ***
            results_with_config.append({'config_obj': config, 'result_data': result_dict})

            success_count += 1

        except Exception as e:
            # Record failure details
            error_msg = f"{type(e).__name__}: {str(e)}"
            # Include traceback for better debugging?
            # import traceback
            # error_msg += f"\n{traceback.format_exc()}"
            failures.append({
                'name': getattr(config, 'name', f'Unnamed Config {i}'),
                'error': error_msg,
                'config_type': config_type,
                'has_specialized_cores': is_specialized,
            })

        finally:
             # Report progress
            if (i + 1) % progress_interval == 0 or (i + 1) == len(configs):
                progress = ((i + 1) / len(configs)) * 100
                current_success_rate = (success_count / (i + 1)) * 100 if (i+1) > 0 else 0
                elapsed_test_time = time.time() - test_start_time_local
                print(f"Completed {i + 1}/{len(configs)} tests ({progress:.1f}%) - "
                      f"Last test took: {elapsed_test_time:.2f}s - "
                      f"Success rate so far: {current_success_rate:.1f}% ({success_count}/{i+1})")


    print("\nTest execution finished.")
    print(f"Successful tests: {success_count}/{len(configs)} ({success_count/len(configs)*100:.1f}%)")
    print(f"Failed tests: {len(failures)}/{len(configs)} ({len(failures)/len(configs)*100:.1f}%)")

    # --- Consolidate and Save Results (Modified) ---
    print("\nConsolidating and saving results...")

    result_records = []
    # *** Iterate through the combined list ***
    for item in results_with_config:
        config = item['config_obj']      # Get the config object
        r = item['result_data']          # Get the result dictionary

        # Determine if this config used specialized core profiles
        is_specialized = hasattr(config, 'core_power_profiles') and config.core_power_profiles

        # Build the flat record for the DataFrame
        record = {
            # Fields from Config object
            'name': config.name,
            'bandwidth_factor': getattr(config, 'bandwidth_factor', None),
            'power_factor': getattr(config, 'power_factor', None),
            'rf_efficiency': getattr(config, 'rf_efficiency', None),
            'time_constraint_multiplier': getattr(config, 'time_constraint_multiplier', None),
            'battery_level': getattr(config, 'battery_level', None),
            'num_cores': getattr(config, 'num_cores', None),
            'edge_nodes': getattr(config, 'num_edge_nodes', 0), # Already in config
            'edge_cores': getattr(config, 'num_edge_cores', 0), # Already in config
            'has_specialized_cores': is_specialized, # Derive from config object

            # Fields directly from result dictionary 'r'
            'initial_time': r.get('initial_time'),
            'final_time': r.get('final_time'),
            'time_constraint': r.get('time_constraint'),
            'initial_energy': r.get('initial_energy'),
            'final_energy': r.get('final_energy'),
            'time_change_percent': r.get('time_change_percent'),
            'energy_reduction_percent': r.get('energy_reduction_percent'),
            'migration_count': r.get('migration_count'),
            'initial_schedule_valid': r.get('initial_schedule_valid'),
            'final_schedule_valid': r.get('final_schedule_valid'),
            'initial_scheduling_duration': r.get('initial_scheduling_duration'),
            'optimization_duration': r.get('optimization_duration'),
            'total_duration': r.get('total_duration'), # Duration of the test run
            'config_type': "specialized" if is_specialized else "standard", # Derive config_type again

            # Fields added for convenience based on distribution dicts
            'initial_local_count': r.get('initial_local_count'),
            'initial_cloud_count': r.get('initial_cloud_count'),
            'final_local_count': r.get('final_local_count'),
            'final_cloud_count': r.get('final_cloud_count'),
            'cloud_migration': r.get('cloud_migration'),
        }
        result_records.append(record)

    # Create the main results DataFrame
    results_df = pd.DataFrame(result_records)

    # Save overall summary CSV only if there are successful results
    if not results_df.empty:
        summary_path = f'{output_dir}/results_summary.csv'
        print(f"Saving results summary ({len(results_df)} rows) to {summary_path}")
        try:
            results_df.to_csv(summary_path, index=False)
        except Exception as e:
            print(f"  ERROR saving summary results CSV: {e}")

        # Check if the column for filtering exists (it should if records were built correctly)
        if 'has_specialized_cores' in results_df.columns:

            # Save specialized results separately if any exist
            if specialized_count > 0: # Check if specialized configs were even generated
                specialized_results_df = results_df[results_df['has_specialized_cores'] == True]
                if not specialized_results_df.empty:
                    spec_path = f'{output_dir}/specialized_results.csv'
                    print(f"Saving specialized results ({len(specialized_results_df)} rows) to {spec_path}")
                    try:
                        specialized_results_df.to_csv(spec_path, index=False)
                    except Exception as e:
                         print(f"  ERROR saving specialized results CSV: {e}")

            # Save standard (sweep) results separately if any exist
            standard_results_df = results_df[results_df['has_specialized_cores'] == False]
            if not standard_results_df.empty:
                std_path = f'{output_dir}/standard_results.csv'
                print(f"Saving standard results ({len(standard_results_df)} rows) to {std_path}")
                try:
                    standard_results_df.to_csv(std_path, index=False)
                except Exception as e:
                    print(f"  ERROR saving standard results CSV: {e}")

        else:
            print("  WARNING: 'has_specialized_cores' column missing in results DataFrame. Check record creation.")

    else:
        print("No successful test results recorded. Skipping results CSV file generation.")

    # --- Save Migration Details ---
    migration_details = []
    # Iterate through the original list which contains the full dicts
    for item in results_with_config:
        r = item['result_data']
        if 'migrations' in r and r.get('migrations'):
             config_name = r.get('config_name', 'Unknown Config') # Get name from result dict
             for m in r['migrations']:
                if isinstance(m, dict):
                    migration_details.append({
                        'config_name': config_name,
                        'task_id': m.get('task_id'),
                        'from_unit': m.get('from_unit'),
                        'to_unit': m.get('to_unit'),
                        'energy_before': m.get('energy_before'),
                        'energy_after': m.get('energy_after'),
                        'time_before': m.get('time_before'),
                        'time_after': m.get('time_after'),
                        'efficiency': m.get('efficiency')
                    })
    # Save migration details if any were collected
    if migration_details:
        try:
            migration_df = pd.DataFrame(migration_details)
            if not migration_df.empty:
                mig_path = f'{output_dir}/migration_details.csv'
                print(f"Saving migration details ({len(migration_df)} rows) to {mig_path}")
                migration_df.to_csv(mig_path, index=False)
        except Exception as e:
            print(f"  ERROR saving migration details CSV: {e}")
    else:
        print("No migration details found to save.")

    # --- Save Failure Details ---
    if failures:
        fail_txt_path = f'{output_dir}/failures.txt'
        fail_csv_path = f'{output_dir}/failures.csv'
        print(f"Saving failure details ({len(failures)} failures) to {fail_txt_path} and {fail_csv_path}")
        try:
            with open(fail_txt_path, 'w') as f:
                f.write(f"Test Suite Failures ({len(failures)}/{len(configs)})\n")
                f.write(f"Run timestamp: {timestamp}\n\n")
                for i, failure in enumerate(failures):
                    f.write(f"Failure #{i+1}: {failure.get('name', 'N/A')} "
                            f"(Type: {failure.get('config_type', 'unknown')}, "
                            f"Specialized: {failure.get('has_specialized_cores', 'N/A')})\n")
                    f.write(f"Error: {failure.get('error', 'No error message')}\n")
                    f.write("-" * 80 + "\n\n")
        except Exception as e: print(f"  ERROR saving failure text file: {e}")
        try:
            failures_df = pd.DataFrame(failures)
            failures_df.to_csv(fail_csv_path, index=False)
        except Exception as e: print(f"  ERROR saving failure CSV file: {e}")
    else:
         print("No failures recorded.")

    # --- Save Overall Test Summary Text File ---
    summary_txt_path = f'{output_dir}/test_summary.txt'
    print(f"Saving overall test summary to {summary_txt_path}")
    try:
        with open(summary_txt_path, 'w') as f:
            f.write(f"Test Suite Summary\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total configurations generated: {len(configs)}\n")
            f.write(f"  Sampling method: {sampling_method}\n")
            if sampling_method == 'random': f.write(f"  Number of samples: {num_samples}\n")
            f.write(f"  Specialized configurations count: {specialized_count}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Execution Results:\n")
            f.write(f"  Successful tests: {success_count}/{len(configs)} ({success_count/len(configs)*100:.1f}%)\n")
            f.write(f"  Failed tests: {len(failures)}/{len(configs)} ({len(failures)/len(configs)*100:.1f}%)\n")
            f.write("-" * 30 + "\n")
            if not results_df.empty:
                avg_energy_reduction = results_df['energy_reduction_percent'].mean()
                avg_time_change = results_df['time_change_percent'].mean()
                avg_migrations = results_df['migration_count'].mean()
                avg_duration = results_df['total_duration'].mean() # Use total_duration from result dict
                f.write(f"Average Metrics (across {len(results_df)} successful tests):\n")
                f.write(f"  - Energy reduction: {avg_energy_reduction:.2f}%\n")
                f.write(f"  - Time change: {avg_time_change:.2f}%\n")
                f.write(f"  - Migrations per config: {avg_migrations:.2f}\n")
                if not pd.isna(avg_duration): f.write(f"  - Avg Total Test Duration: {avg_duration:.3f}s\n")
            else: f.write("Average Metrics: Not calculated (no successful tests).\n")
    except Exception as e:
        print(f"  ERROR saving test summary text file: {e}")

    print("\nTest suite finished.")
    return results_df, failures, output_dir


if __name__ == '__main__':
    print("--- Running Specific MCC Configuration Tests ---")


    target_config_names = [
        "Local-Favoring_Cores_3",
        "Cloud-Favoring_BW_2.0",
        "Battery-Critical_15pct",
    ]
    print(f"Attempting to select and run {len(target_config_names)} specific configurations by name.")
    print(f"Target names: {target_config_names}")

    print("Generating the set of predefined specialized configurations...")
    all_specialized_configs = generate_configs(param_ranges=None, seed=42)
    print(f"Found {len(all_specialized_configs)} predefined specialized configurations in total.")

    # Filter the generated specialized configs to get only the ones we specifically want to test
    configs_to_test = [cfg for cfg in all_specialized_configs if cfg.name in target_config_names]

    # --- Validation and Execution ---
    if not configs_to_test:
        print("\n*** WARNING: No target configurations found among the generated specialized set! ***")
        print("*** Check the names in `target_config_names` against the actual names defined ")
        print("*** inside the `data.generate_configs` function. Execution cannot proceed. ***\n")
    elif len(configs_to_test) < len(target_config_names):
         found_names = {cfg.name for cfg in configs_to_test}
         missing_names = [name for name in target_config_names if name not in found_names]
         print(f"\n*** WARNING: Could not find the following target configurations: {missing_names}. ***")
         print(f"*** Will proceed testing the {len(configs_to_test)} configurations that were found. ***\n")
    else:
         print(f"Successfully selected all {len(configs_to_test)} target configurations.")


    # Proceed only if we found configurations to test
    results_list = []
    failures_list = []
    if configs_to_test:
        print(f"\n--- Starting execution of {len(configs_to_test)} selected tests ---")

        for i, config in enumerate(configs_to_test):
            print(f"\n--- Test {i+1}/{len(configs_to_test)} ---")
            try:
                # Run the test for the specific configuration
                result = run_unified_test(config)
                results_list.append(result)
                print(f"--- Test '{config.name}' Completed Successfully ---")

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"!!! Test '{config.name}' Failed: {type(e).__name__} !!!")
                print(error_msg) # Print traceback for debugging
                failures_list.append({'config_name': config.name, 'error': error_msg})

    # --- Display Summary of Results ---
    print("\n" + "=" * 35 + " Specific Test Run Summary " + "=" * 35) # Adjusted spacing
    print(f"Total configurations targeted: {len(target_config_names)}")
    print(f"Total configurations found/run: {len(configs_to_test)}")
    print(f"Successful tests: {len(results_list)}")
    print(f"Failed tests: {len(failures_list)}")

    if results_list:
        print("\nKey Metrics from Successful Runs:")
        # Updated header to include T_max
        print("-" * 98) # Adjusted width
        print(f"{'Config Name':<30} | {'T_init':>8} | {'T_final':>8} | {'T_max':>8} | {'E_init':>10} | {'E_final':>10} | {'Valid':>5} | {'Migr':>4}")
        print("-" * 98) # Adjusted width
        for r in results_list:
            valid_str = "OK" if r.get('final_schedule_valid', False) else "FAIL"
            migr_count = r.get('migration_count', 'N/A')
            # Added T_max (time_constraint from results)
            t_max_val = r.get('time_constraint', 0)
            print(f"{r.get('config_name', 'N/A'):<30} | {r.get('initial_time', 0):>8.2f} | {r.get('final_time', 0):>8.2f} | {t_max_val:>8.2f} | "
                  f"{r.get('initial_energy', 0):>10.2f} | {r.get('final_energy', 0):>10.2f} | {valid_str:>5} | {migr_count:>4}")
        print("-" * 98) # Adjusted width

        # Add a check for T_final exceeding T_max
        violation_count = 0
        for r in results_list:
            if r.get('final_time', 0) > r.get('time_constraint', float('inf')) + 1e-9: # Add tolerance
                 print(f"  WARNING: Config '{r.get('config_name')}' - Final Time {r.get('final_time'):.2f} exceeds T_max {r.get('time_constraint'):.2f}")
                 violation_count += 1
        if violation_count == 0:
            print("  All successful runs met their T_max constraint.")
        else:
            print(f"  {violation_count} successful run(s) potentially violated the T_max constraint.")


    if failures_list:
        print("\nFailures Encountered:")
        for f in failures_list:
            print(f" - {f['config_name']}: {f['error'].splitlines()[0]}") # Print first line of error

    print("\n--- End of Specific Test Run ---")