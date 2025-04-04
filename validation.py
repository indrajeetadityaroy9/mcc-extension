#validation.py
from data import ExecutionTier, SchedulingState

def validate_task_dependencies_no_edge(tasks, epsilon=1e-9):
    """
    Verifies task dependencies for a two-tier (Device/Cloud) schedule,
    aligning strictly with the ready time definitions in the Lin et al. 2014 paper.

    - For DEVICE successors: Checks against Eq (3) - predecessor results must be
      fully available on device (max(FT_l, FT_wr)).
    - For CLOUD successors: Checks against Eq (4) - predecessor must have finished
      locally (FT_l) or finished *sending* (FT_ws).

    Args:
        tasks: List of Task objects with scheduling information populated.
        epsilon: Float tolerance for floating point comparisons.

    Returns:
        (is_valid, violations): Tuple where:
          - is_valid: bool, True if no violations
          - violations: list of dicts describing each violation
    """
    violations = []
    task_map = {task.id: task for task in tasks} # Quick lookup

    for task in tasks:
        # Skip tasks that weren't scheduled or assigned
        if task.is_scheduled == SchedulingState.UNSCHEDULED or task.assignment == -2:
            continue

        # Skip tasks explicitly assigned to Edge (as this validator is two-tier)
        if hasattr(task, 'execution_tier') and task.execution_tier == ExecutionTier.EDGE:
             continue

        # --- Determine Successor Task's Effective Start Time ---
        task_start_time = -1.0
        task_location_str = ""
        task_critical_start_event = "" # Description of what the start time represents

        if not hasattr(task, 'execution_tier') or task.execution_tier is None:
            violations.append({
                'type': 'Missing Execution Tier', 'task': task.id,
                'detail': f"Task {task.id} is scheduled but missing execution_tier."})
            continue

        if task.execution_tier == ExecutionTier.DEVICE:
            core_id = getattr(task, 'device_core', -1)
            if task.execution_unit_task_start_times and 0 <= core_id < len(task.execution_unit_task_start_times):
                task_start_time = task.execution_unit_task_start_times[core_id]
                # Ensure start time is non-negative
                if task_start_time < 0:
                    violations.append({
                        'type': 'Invalid Start Time', 'task': task.id,
                        'detail': f"Task {task.id} on DEVICE Core {core_id} has negative start time {task_start_time:.3f}."})
                    continue
                task_location_str = f"Device Core {core_id}"
                task_critical_start_event = "local execution"
            else:
                violations.append({
                    'type': 'Invalid Device Assignment/Timing', 'task': task.id,
                    'detail': f"Task {task.id} on DEVICE has invalid core {core_id} or missing/malformed start times array."})
                continue

        elif task.execution_tier == ExecutionTier.CLOUD:
            # 'assignment' should be the cloud index (e.g., num_cores)
            cloud_idx = task.assignment
            if task.execution_unit_task_start_times and 0 <= cloud_idx < len(task.execution_unit_task_start_times):
                task_start_time = task.execution_unit_task_start_times[cloud_idx]
                # Ensure start time is non-negative
                if task_start_time < 0:
                    violations.append({
                        'type': 'Invalid Start Time', 'task': task.id,
                        'detail': f"Task {task.id} on CLOUD has negative upload start time {task_start_time:.3f}."})
                    continue
                task_location_str = "Cloud"
                task_critical_start_event = "upload start" # The critical event for dependencies
            else:
                 violations.append({
                    'type': 'Invalid Cloud Assignment/Timing', 'task': task.id,
                    'detail': f"Task {task.id} on CLOUD has invalid assignment index {task.assignment} or missing/malformed start times array."})
                 continue
        else:
             # Should have been skipped earlier, but added for completeness
             violations.append({
                 'type': 'Unsupported Tier', 'task': task.id,
                 'detail': f"Task {task.id} has unsupported execution tier {task.execution_tier.name} for validation."})
             continue

        # --- Check Each Predecessor ---
        for pred_task_ref in task.pred_tasks:
            pred_task = task_map.get(pred_task_ref.id)

            # Check if predecessor exists and was scheduled
            if not pred_task:
                 violations.append({
                    'type': 'Missing Predecessor Object', 'task': task.id, 'predecessor': pred_task_ref.id,
                    'detail': f"Could not find predecessor task object {pred_task_ref.id} in task map."})
                 continue
            if pred_task.is_scheduled == SchedulingState.UNSCHEDULED or pred_task.assignment == -2:
                violations.append({
                    'type': 'Unscheduled Predecessor', 'task': task.id, 'predecessor': pred_task.id,
                    'detail': f"Task {task.id} depends on unscheduled predecessor {pred_task.id}."})
                continue

            # Skip checking dependencies *from* Edge tasks
            if hasattr(pred_task, 'execution_tier') and pred_task.execution_tier == ExecutionTier.EDGE:
                continue

            if not hasattr(pred_task, 'execution_tier') or pred_task.execution_tier is None:
                 violations.append({
                     'type': 'Missing Execution Tier', 'task': task.id, 'predecessor': pred_task.id,
                     'detail': f"Predecessor task {pred_task.id} is scheduled but missing execution_tier."})
                 continue

            # --- Determine Required Predecessor Availability Time ---
            # This depends on WHERE the CURRENT task (`task`) is executing.
            pred_available_time = -1.0
            pred_required_event = "" # Description of what needs to finish for the predecessor

            # Get predecessor's relevant finish times, defaulting to 0 if not applicable or missing
            # A finish time of 0 generally means the task didn't execute via that path.
            pred_ft_l = getattr(pred_task, 'FT_l', 0)
            pred_ft_ws = getattr(pred_task, 'FT_ws', 0)
            pred_ft_wr = getattr(pred_task, 'FT_wr', 0)
            
            # Ensure finish times used for max are non-negative
            pred_ft_l = max(0, pred_ft_l)
            pred_ft_ws = max(0, pred_ft_ws)
            pred_ft_wr = max(0, pred_ft_wr)

            if task.execution_tier == ExecutionTier.DEVICE:
                # **Rule based on Eq (3):** Device execution requires results fully available.
                pred_available_time = max(pred_ft_l, pred_ft_wr)
                pred_required_event = "local finish (FT_l)" if pred_ft_l >= pred_ft_wr else "cloud result receive (FT_wr)"
            elif task.execution_tier == ExecutionTier.CLOUD:
                # **Rule based on Eq (4):** Cloud upload requires local completion OR cloud send completion.
                pred_available_time = max(pred_ft_l, pred_ft_ws)
                pred_required_event = "local finish (FT_l)" if pred_ft_l >= pred_ft_ws else "cloud send finish (FT_ws)"

            # If available time is still negative, something is wrong (shouldn't happen with max(0, ...))
            if pred_available_time < 0:
                 violations.append({
                    'type': 'Internal Validation Error', 'task': task.id, 'predecessor': pred_task.id,
                    'detail': f"Could not determine valid predecessor available time for {pred_task.id}."})
                 continue


            # --- Perform the Comparison ---
            time_diff = task_start_time - pred_available_time

            # Violation if predecessor finishes (or is available) significantly AFTER task starts
            # Use math.isclose for near-zero checks if needed, but simple comparison is often sufficient
            # if pred_available_time > task_start_time + epsilon:
            # Or check if time_diff is significantly negative:
            if time_diff < -epsilon:
                # Violation!
                violation_type = f"Dependency Violation ({pred_task.execution_tier.name} -> {task.execution_tier.name})"
                violations.append({
                    'type': violation_type,
                    'task': task.id,
                    'predecessor': pred_task.id,
                    'task_start_time': round(task_start_time, 4),
                    'pred_available_time': round(pred_available_time, 4),
                    'detail': f"Task {task.id} ({task_critical_start_event}) starts at {task_start_time:.4f} "
                              f"but predecessor {pred_task.id} is only available (via {pred_required_event}) at {pred_available_time:.4f}."
                })

    is_valid = (len(violations) == 0)
    return is_valid, violations

def validate_task_dependencies(tasks, epsilon=1e-9):
    """
    Verifies that each scheduled task starts only after its immediate predecessors'
    results are available at the location where the task needs them (device, edge, or cloud),
    considering the three-tier architecture.

    Args:
        tasks: List of Task objects with scheduling information populated
               (execution_tier, assignment, FT_l, FT_ws, FT_c, FT_wr, FT_edge, FT_edge_receive,
                execution_unit_task_start_times, pred_tasks, etc.)
        epsilon: float tolerance for floating point comparisons

    Returns:
        (is_valid, violations): Tuple where:
          - is_valid: bool, True if no violations
          - violations: list of dicts describing each violation
    """
    violations = []
    task_map = {task.id: task for task in tasks} # Quick lookup

    for task in tasks:
        # Skip tasks that weren't scheduled
        if task.is_scheduled == SchedulingState.UNSCHEDULED or task.assignment == -2:
            continue

        # Determine the actual start time of the current task based on its tier
        task_start_time = -1.0
        task_location_str = ""

        if task.execution_tier == ExecutionTier.DEVICE:
            core_id = task.device_core
            if 0 <= core_id < len(task.execution_unit_task_start_times):
                task_start_time = task.execution_unit_task_start_times[core_id]
                task_location_str = f"Device Core {core_id}"
            else: # Inconsistency
                violations.append({
                    'type': 'Invalid Device Assignment', 'task': task.id,
                    'detail': f"Task {task.id} on DEVICE has invalid core {core_id}"})
                continue

        elif task.execution_tier == ExecutionTier.CLOUD:
            cloud_idx = task.assignment # Should match num_cores
            if 0 <= cloud_idx < len(task.execution_unit_task_start_times):
                 # For cloud, the "start" relevant for dependencies is the upload start
                task_start_time = task.execution_unit_task_start_times[cloud_idx]
                task_location_str = "Cloud (Upload Start)"
            else: # Inconsistency
                violations.append({
                    'type': 'Invalid Cloud Assignment', 'task': task.id,
                    'detail': f"Task {task.id} on CLOUD has invalid assignment {task.assignment}"})
                continue

        elif task.execution_tier == ExecutionTier.EDGE:
            if task.edge_assignment:
                e_id, c_id = task.edge_assignment.edge_id, task.edge_assignment.core_id
                edge_unit_idx = task.assignment # Should correspond to (e_id, c_id)
                if 0 <= edge_unit_idx < len(task.execution_unit_task_start_times):
                    # The relevant start time is the edge execution start
                    task_start_time = task.execution_unit_task_start_times[edge_unit_idx]
                    task_location_str = f"Edge ({e_id},{c_id})"
                else: # Inconsistency
                    violations.append({
                        'type': 'Invalid Edge Assignment', 'task': task.id,
                        'detail': f"Task {task.id} on EDGE has invalid assignment index {edge_unit_idx}"})
                    continue
            else: # Inconsistency
                 violations.append({
                        'type': 'Missing Edge Assignment', 'task': task.id,
                        'detail': f"Task {task.id} on EDGE is missing edge_assignment details"})
                 continue

        # If start time couldn't be determined, skip dependency check for this task
        if task_start_time < 0:
            violations.append({
                'type': 'Missing Start Time', 'task': task.id,
                'detail': f"Could not determine valid start time for Task {task.id} on {task.execution_tier.name}"})
            continue


        # Check each predecessor
        for pred_task_obj in task.pred_tasks:
            # Find the predecessor task object if needed (safer than assuming list contains objects)
            pred_task = task_map.get(pred_task_obj.id)
            if not pred_task or pred_task.is_scheduled == SchedulingState.UNSCHEDULED:
                violations.append({
                    'type': 'Unscheduled Predecessor', 'task': task.id, 'predecessor': pred_task_obj.id,
                    'detail': f"Task {task.id} depends on unscheduled task {pred_task_obj.id}"})
                continue

            # Determine when the predecessor's results are available *where needed* by the current task
            pred_finish_time_needed = -1.0

            # Case 1: Current task starts on DEVICE
            if task.execution_tier == ExecutionTier.DEVICE:
                # Needs results available AT THE DEVICE
                pred_finish_time_needed = pred_task.get_final_finish_time()

            # Case 2: Current task starts upload to CLOUD
            elif task.execution_tier == ExecutionTier.CLOUD:
                 # Needs results available AT THE DEVICE to start upload
                 pred_finish_time_needed = pred_task.get_final_finish_time()

            # Case 3: Current task starts execution on EDGE
            elif task.execution_tier == ExecutionTier.EDGE:
                 # Needs results available AT THE DEVICE to start upload to edge
                 # (because edge exec start depends on upload finish)
                 pred_finish_time_needed = pred_task.get_final_finish_time()


            # Check for violation
            if pred_finish_time_needed < 0:
                 violations.append({
                    'type': 'Missing Predecessor Finish Time', 'task': task.id, 'predecessor': pred_task.id,
                    'detail': f"Could not determine finish time for predecessor {pred_task.id} ({pred_task.execution_tier.name}) needed by task {task.id}"})
                 continue

            if (pred_finish_time_needed - task_start_time) > epsilon:
                 # Violation! Predecessor finishes after task starts
                 violations.append({
                     'type': f'Dependency Violation ({pred_task.execution_tier.name} -> {task.execution_tier.name})',
                     'task': task.id,
                     'predecessor': pred_task.id,
                     'detail': f"Task {task.id} starts at {task_start_time:.3f} on {task_location_str}, "
                               f"but predecessor {pred_task.id} results available at {pred_finish_time_needed:.3f}"
                 })

    is_valid = (len(violations) == 0)
    return is_valid, violations