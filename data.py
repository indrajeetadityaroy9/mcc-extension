#data.py

import random
import itertools
import numpy as np
from enum import Enum
from dataclasses import dataclass

class ExecutionTier(Enum):
    """
    ExecutionTier defines where a task can be executed in the three-tier architecture:
      - DEVICE: Execution on the mobile device (using local cores).
      - EDGE: Execution on edge nodes (intermediate tier).
      - CLOUD: Execution on the cloud platform.
    """
    DEVICE = 0  # Mobile device (local cores)
    EDGE = 1  # Edge nodes (intermediate tier)
    CLOUD = 2  # Cloud platform


class SchedulingState(Enum):
    """
    SchedulingState defines the state of task scheduling in the algorithm:
      - UNSCHEDULED: Task has not been scheduled.
      - SCHEDULED: Task has been scheduled during the initial minimal-delay scheduling.
      - KERNEL_SCHEDULED: Task has been rescheduled after energy optimization.
    """
    UNSCHEDULED = 0  # Initial state
    SCHEDULED = 1  # After initial minimal-delay scheduling
    KERNEL_SCHEDULED = 2  # After energy optimization

@dataclass
class TaskMigrationState:
    """
    Class to track task migration decisions and their outcomes,
    as described conceptually in Section III.B of the reference paper.
    Used to evaluate the effectiveness of a potential task migration.
    """
    time: float  # T_total: Total application completion time after the potential migration
    energy: float  # E_total: Total mobile device energy consumption after the potential migration
    efficiency: float  # Metric to compare migrations (e.g., energy reduction / time increase)
    task_index: int  # v_tar: The index (ID) of the task considered for migration
    target_execution_unit: int  # k_tar: Target unit index (core index, or indicator for cloud/edge)

# --- Global Variables ---
core_execution_times = {}   # Stores {task_id: [list of times on each device core]}
cloud_execution_times = {}  # Stores {task_id: [send_time, compute_time, receive_time]}
edge_execution_times = {}   # Stores {task_id: {(edge_id, core_id): time}}

def generate_mcc_power_models(device_type='mobile', battery_level=100.0, power_factor=1.0, rf_efficiency=1.0, num_cores=3, num_edge_nodes=1, num_edge_cores=1, core_power_profiles=None, seed=None):
    """
    Generates customizable power consumption models for different components
    in the MCC environment (device cores, edge cores, cloud, RF communication).

    Args:
        device_type (str): Type of the primary device ('mobile', etc.). Currently mainly affects default profiles.
        battery_level (float): Current battery percentage (influences mobile device power).
        power_factor (float): A global multiplier for all power consumption values.
        rf_efficiency (float): Efficiency factor for Radio Frequency components (lower means more power).
        num_cores (int): Number of cores in the mobile device.
        num_edge_nodes (int): Number of edge computing nodes available.
        num_edge_cores (int): Number of cores within each edge node.
        core_power_profiles (list, optional): Predefined list of dictionaries describing power characteristics for each mobile device core. If None, default profiles are used.
        seed (int, optional): Random seed for reproducibility in variations.

    Returns:
        dict: A dictionary containing power models structured by component type:
              {'device': {core_id: model}, 'edge': {(edge_id, core_id): model}, 'cloud': model, 'rf': {link_type: model}}
              Models typically include 'idle_power' and a 'dynamic_power' lambda function.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Initialize dictionary to store power models for each component type
    power_models = {
        'device': {},  # Power models for mobile device cores
        'edge': {},    # Power models for edge node cores
        'cloud': {},   # Power model for the cloud
        'rf': {}       # Power models for various communication links
    }

    # --- Mobile Device Power ---
    if device_type == 'mobile' or num_cores > 0: # Generate device models if cores exist
        # Calculate battery factor: low battery slightly increases power consumption
        battery_factor = 1.0 if battery_level > 30 else 1.0 + (30 - battery_level) * 0.01

        # Use provided core profiles or generate default ones based on num_cores
        if core_power_profiles:
            # Use provided profiles, ensuring the list matches num_cores
            core_power_profiles = core_power_profiles[:num_cores]
        else:
            # Define default core profiles for a mobile device (example: big.LITTLE style)
            core_power_profiles = [
                {  # High-performance core (e.g., "big")
                    'idle_power': 0.1,
                    'power_coefficient': 1.8, # Higher coefficient means more power per unit load
                    'power_base': 0.2,        # Base active power
                    'frequency_range': (0.8, 2.4), # GHz
                    'current_frequency': 2.0,
                },
                {  # Mid-range core
                    'idle_power': 0.05,
                    'power_coefficient': 1.4,
                    'power_base': 0.1,
                    'frequency_range': (0.6, 1.8),
                    'current_frequency': 1.6,
                },
                {  # Efficiency core (e.g., "LITTLE")
                    'idle_power': 0.03,
                    'power_coefficient': 0.95,
                    'power_base': 0.05,
                    'frequency_range': (0.5, 1.5),
                    'current_frequency': 1.2,
                }
            ]

            # Adjust the default core list to match the required num_cores
            while len(core_power_profiles) < num_cores:
                # If more cores are needed, duplicate the last (most efficient) profile
                core_power_profiles.append(core_power_profiles[-1].copy())
            # Trim the list if num_cores is less than the number of default profiles
            core_power_profiles = core_power_profiles[:num_cores]

        # Generate the power model dictionary for each mobile core
        for i, profile in enumerate(core_power_profiles):
            # Apply battery and global power factors
            idle = profile['idle_power'] * battery_factor * power_factor
            coef = profile['power_coefficient'] * power_factor
            base = profile['power_base'] * power_factor

            power_models['device'][i] = {
                'idle_power': idle,
                # Lambda function to compute dynamic power based on current load (0 to 1)
                # Power = (Base + Coefficient * Load) * BatteryFactor
                'dynamic_power': lambda load, c=coef, b=base, bf=battery_factor: (b + c * load) * bf,
                'frequency_range': profile['frequency_range'],
                'current_frequency': profile['current_frequency'],
            }

        # --- RF Transmission Power Models ---
        # Models estimate power based on data rate and potentially signal strength.
        # These models are simplified approximations.
        # Sending power: Higher data rate or weaker signal generally means more power.
        device_to_edge_power = lambda data_rate, signal_strength=70, bf=battery_factor, rf_eff=rf_efficiency: (
            # Base power + power proportional to rate and inverse signal strength
            (0.1 + 0.4 * (data_rate / 10.0) * (1 + (70 - signal_strength) * 0.02)) * bf / rf_eff
        )
        device_to_cloud_power = lambda data_rate, signal_strength=70, bf=battery_factor, rf_eff=rf_efficiency: (
            # Sending to cloud typically requires more power than to edge
            (0.15 + 0.6 * (data_rate / 5.0) * (1 + (70 - signal_strength) * 0.03)) * bf / rf_eff
        )

        # Receiving power (lower than sending)
        edge_to_device_power = lambda data_rate, signal_strength=70, rf_eff=rf_efficiency: (
             (0.05 + 0.1 * (data_rate / 12.0)) / rf_eff
        )
        # Edge RF Power Models
        edge_to_cloud_power = lambda data_rate, rf_eff=rf_efficiency: ( # Edge sending to cloud
             (0.2 + 0.8 * (data_rate / 50.0)) / rf_eff
        )
        cloud_to_edge_power = lambda data_rate, rf_eff=rf_efficiency: ( # Edge receiving from cloud
             (0.1 + 0.2 * (data_rate / 60.0)) / rf_eff
        )
        edge_to_edge_power = lambda data_rate, rf_eff=rf_efficiency: ( # Edge to edge communication
             (0.15 + 0.5 * (data_rate / 30.0)) / rf_eff
        )

        # Store RF power models in the dictionary
        power_models['rf'] = {
            'device_to_edge': device_to_edge_power,
            'device_to_cloud': device_to_cloud_power,
            'edge_to_device': edge_to_device_power,
            'edge_to_cloud': edge_to_cloud_power,
            'cloud_to_edge': cloud_to_edge_power,
            'edge_to_edge': edge_to_edge_power,
        }

    # --- Edge Server Power ---
    if num_edge_nodes > 0 and num_edge_cores > 0:
        # Create power models for each core within each edge node
        for edge_id in range(1, num_edge_nodes + 1):
            for core_id in range(1, num_edge_cores + 1):
                # Introduce heterogeneity based on edge/core ID (e.g., older/less efficient nodes)
                # Calculate a base efficiency factor that decreases for higher IDs
                base_efficiency = 1.0 - 0.05 * (edge_id - 1) - 0.02 * (core_id - 1)
                efficiency_variation = random.uniform(0.9, 1.1) # Add random variation per core
                efficiency = max(0.5, base_efficiency * efficiency_variation) # Ensure minimum efficiency

                # Power multiplier is inverse of efficiency, adjusted by global power factor
                power_multiplier = 1.0 / efficiency * power_factor

                edge_core_key = (edge_id, core_id) # Use tuple as key
                power_models['edge'][edge_core_key] = {
                    'idle_power': 5.0 * power_multiplier, # Example base idle power for edge core
                    # Dynamic power model for edge core, scaled by the multiplier
                    'dynamic_power': lambda load, mult=power_multiplier: (3.0 + 12.0 * load) * mult,
                    'frequency_range': (1.0, 3.2), # Example frequency range
                    'current_frequency': 2.8,      # Example current frequency
                    # Store the multiplier; can be used to influence execution time (less efficient -> slower)
                    'nodeFactor': power_multiplier
                }

    # --- Cloud Server Power ---
    # Define a simple, generally powerful model for cloud servers (representing aggregate power)
    power_models['cloud'] = {
        'idle_power': 50.0 * power_factor, # High idle power representing underlying infrastructure
        'dynamic_power': lambda load, pf=power_factor: (20.0 + 180.0 * load) * pf, # High dynamic range
        'frequency_range': (2.0, 4.0),
        'current_frequency': 3.5 # High operating frequency
    }

    # Return the complete dictionary of power models
    return power_models


def generate_mcc_network_conditions(bandwidth_factor=1.0, seed=None):
    """
    Generates network bandwidth conditions (upload/download rates) for various links
    in the three-tier MCC architecture (Device, Edge, Cloud).

    Args:
        bandwidth_factor (float): A multiplier to scale all base bandwidth rates.
                                  > 1 means better network, < 1 means worse network.
        seed (int, optional): Random seed for potential future extensions (e.g., jitter).

    Returns:
        tuple: A tuple containing two dictionaries:
               (upload_rates, download_rates)
               Each dictionary maps link types (str) to bandwidth rates (float, in Mbps).
    """
    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Define base upload rates (in Mbps) for different network links.
    base_upload = {
        'device_to_edge': 10.0,  # Mobile device (uplink) to edge node.
        'edge_to_edge': 30.0,    # Between two edge nodes (symmetric assumed here).
        'edge_to_cloud': 50.0,   # Edge node (uplink) to cloud.
        'device_to_cloud': 5.0,  # Mobile device (uplink) directly to cloud (often slower).
    }

    # Define base download rates (in Mbps) for different network links.
    base_download = {
        'edge_to_device': 12.0,   # Edge node (downlink) to mobile device.
        'cloud_to_edge': 60.0,    # Cloud (downlink) to edge node.
        'edge_to_edge': 30.0,     # Between edge nodes (symmetric).
        'cloud_to_device': 6.0,   # Cloud (downlink) to mobile device.
    }

    # Scale the base upload and download rates by the bandwidth_factor.
    upload_rates = {link: rate * bandwidth_factor for link, rate in base_upload.items()}
    download_rates = {link: rate * bandwidth_factor for link, rate in base_download.items()}

    # Return the scaled upload and download rates.
    return upload_rates, download_rates

def generate_edge_task_execution_times(tasks, mcc_edge_power_models, num_edge_nodes, num_edge_cores, alpha_local_cloud=0.7, seed=None):
    """
    Calculates and assigns estimated execution times for each task on each available edge core.
    The edge execution time is modeled as a blend of the task's minimum local execution time
    and its total cloud execution time (including transfers), adjusted by the specific
    edge node/core efficiency factor and task characteristics.

    Args:
        tasks (list): List of task objects (expected to have attributes like 'id',
                      'local_execution_times', 'cloud_execution_times', potentially 'task_type', etc.).
                      Task objects will be modified in-place to add 'edge_execution_times'.
        mcc_edge_power_models (dict): Dictionary containing power models for edge cores,
                                      specifically used here to retrieve the 'nodeFactor'
                                      which reflects relative efficiency.
        num_edge_nodes (int): Number of edge nodes.
        num_edge_cores (int): Number of cores per edge node.
        alpha_local_cloud (float): Weight determining the blend between local and cloud performance
                                   to estimate base edge performance (0.0 to 1.0).
                                   0.7 means 70% based on local time, 30% based on cloud time.
        seed (int, optional): Random seed for reproducibility of random variations.

    Returns:
        None: Modifies the task objects in the input list `tasks` by adding/updating
              the `edge_execution_times` dictionary attribute. Also updates the global
              `edge_execution_times` dictionary.
    """
    # Access global variables containing previously generated times.
    global core_execution_times, cloud_execution_times
    if seed is not None:
        random.seed(seed)

    for task_obj in tasks:
        # Get the task ID, handling potential missing attribute.
        task_id = getattr(task_obj, 'id', None)
        if task_id is None:
            print(f"Warning: Task object missing 'id' attribute in generate_edge_task_execution_times.")
            continue

        # --- Determine Base Edge Time ---
        # 1. Get minimum local execution time for this task.
        # Try fetching from globals first, then fallback to task attribute.
        local_times = core_execution_times.get(task_id, [])
        if not local_times:
            local_times = getattr(task_obj, 'local_execution_times', []) # Fallback
        min_local_time = min(local_times) if local_times else 10.0 # Provide a default if no times found

        # 2. Get total cloud execution time (send + compute + receive).
        # Try fetching from globals first, then fallback to task attribute.
        cloud_phase_times = cloud_execution_times.get(task_id, [])
        if not cloud_phase_times:
             cloud_phase_times = getattr(task_obj, 'cloud_execution_times', []) # Fallback
        # Ensure we have all three phases before summing
        total_cloud_time = sum(cloud_phase_times) if len(cloud_phase_times) >= 3 else 5.0 # Provide a default

        # 3. Calculate the base edge time using a weighted blend.
        # This approximates edge performance relative to device and cloud.
        base_edge_time = alpha_local_cloud * min_local_time + (1.0 - alpha_local_cloud) * total_cloud_time
        base_edge_time = max(1.0, base_edge_time) # Ensure a minimum sensible execution time

        # 4. Apply task-specific adjustment
        # Adjust the base time based on task type, complexity, data intensity if available.
        adjustment = 1.0
        if hasattr(task_obj, 'task_type') and hasattr(task_obj, 'complexity'):
            task_type = str(getattr(task_obj, 'task_type', 'balanced')).lower()
            complexity = getattr(task_obj, 'complexity', 1.0)
            data_intensity = getattr(task_obj, 'data_intensity', 1.0)

            if task_type == 'compute':
                # Compute-intensive tasks might be relatively slower on edge (compared to cloud)
                # Increase time slightly based on complexity.
                adjustment *= (1.0 + 0.1 * (complexity / 3.0)) # Scale up time
            elif task_type == 'data':
                # Data-intensive tasks might benefit more from edge proximity (reduced latency).
                # Decrease time slightly based on data intensity.
                adjustment *= (1.0 - 0.1 * (data_intensity / 2.0)) # Scale down time

            adjustment = max(0.8, min(adjustment, 1.2)) # Bound the adjustment factor

        adjusted_base_time = base_edge_time * adjustment

        # --- Assign Time for Each Specific Edge Core ---
        # Ensure the task object has the dictionary attribute to store results.
        if not hasattr(task_obj, 'edge_execution_times'):
            task_obj.edge_execution_times = {}

        # Iterate through all available edge cores.
        for edge_id in range(1, num_edge_nodes + 1):
            for core_id in range(1, num_edge_cores + 1):
                edge_core_key = (edge_id, core_id)

                # Get the node/core specific efficiency factor from its power model.
                edge_core_model = mcc_edge_power_models.get(edge_core_key, {})
                # 'nodeFactor' sets efficiency (higher factor means less efficient -> longer time).
                node_factor = edge_core_model.get('nodeFactor', 1.0) # Default to 1.0 if not found

                # Calculate final execution time for this task on this specific edge core.
                # Base time adjusted by task characteristics, scaled by node efficiency factor.
                computed_time = adjusted_base_time * node_factor
                # Add a small random variation per core for more realism.
                computed_time *= random.uniform(0.95, 1.05)
                computed_time = max(0.5, computed_time) # Ensure a minimum execution time

                # Store the computed time in the task's dictionary.
                task_obj.edge_execution_times[edge_core_key] = computed_time

                # print(f"  Task {task_id}, Edge ({edge_id},{core_id}): Time = {computed_time:.2f} (Base: {base_edge_time:.2f}, AdjBase: {adjusted_base_time:.2f}, Factor: {node_factor:.2f})")

    # Update the global dictionary
    global edge_execution_times
    edge_execution_times = {task.id: task.edge_execution_times for task in tasks if hasattr(task, 'id') and hasattr(task, 'edge_execution_times')}


def generate_task_execution_times(num_tasks, num_cores, local_time_range=(1, 15), cloud_phases_range=(0.5, 3.0), heterogeneity_factor=0.3, variation_factor=0.2, nonlinear_prob=0.3, compute_intensity_range=(0.5, 2.0), data_size_range=(0.5, 3.0), seed=None):
    """
    Generates estimated execution times for tasks on heterogeneous local device cores
    and estimates the time breakdown (send, compute, receive) for cloud execution.

    Args:
        num_tasks (int): Number of tasks to generate times for.
        num_cores (int): Number of cores in the mobile device.
        local_time_range (tuple): (min, max) range for base execution time on the fastest local core.
        cloud_phases_range (tuple): (min, max) range for base time factor of cloud transfer phases.
        heterogeneity_factor (float): Controls the speed difference between the fastest and slowest core.
                                     0 means all cores are same speed, 1 means slowest is near zero speed.
        variation_factor (float): Random multiplicative factor range (1 +/- variation_factor) applied to times.
        nonlinear_prob (float): Probability (0-1) that a task exhibits nonlinear speedup across cores
                                (e.g., diminishing returns on more cores).
        compute_intensity_range (tuple): (min, max) range for task computational intensity factor.
                                         Higher values mean task is more compute-bound.
        data_size_range (tuple): (min, max) range for task data size factor. Affects cloud transfer times.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (core_times, cloud_times)
               core_times (dict): {task_id: [list of execution times on each core]}
               cloud_times (dict): {task_id: [send_time, compute_time, receive_time] for cloud}
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    core_times = {}   # Store per-task execution times on local cores.
    cloud_times = {}  # Store per-task cloud phase times [send, compute, receive].

    # Generate execution times for each task.
    for task_id in range(1, num_tasks + 1):
        # Determine a base execution time for the task (on the fastest core concept).
        base_time = random.uniform(local_time_range[0], local_time_range[1])
        # Decide if the task will exhibit nonlinear speedup across cores.
        nonlinear = random.random() < nonlinear_prob

        # Determine task characteristics affecting cloud times.
        compute_intensity = random.uniform(compute_intensity_range[0], compute_intensity_range[1])
        data_size = random.uniform(data_size_range[0], data_size_range[1])

        core_times[task_id] = []  # Initialize list for this task's core times.

        # Generate execution times for each local core based on linearity and heterogeneity.
        if nonlinear:
            # Nonlinear speedup: Performance gain diminishes on slower/more cores.
            core_times[task_id].append(base_time) # Time on the first (fastest) core
            for core_idx in range(1, num_cores):
                # Apply a speed reduction factor based on core index (quadratic slowdown)
                # Slower cores take proportionally longer than the linear model predicts.
                speedup_factor = 1.0 - heterogeneity_factor * (core_idx / max(1, num_cores - 1))**1.5
                # Calculate time, apply random variation, ensure minimum time
                time = max(1.0, base_time * speedup_factor * random.uniform(1 - variation_factor, 1 + variation_factor))
                # Add a small chance of stalling - time similar to the previous core
                if random.random() < 0.10: # 10% chance
                    time = core_times[task_id][-1] * random.uniform(0.95, 1.05) # Take time similar to previous core
                core_times[task_id].append(max(1.0, time)) # Ensure minimum time
        else:
            # Linear speedup: Performance scales linearly with core speed
            for core_idx in range(num_cores):
                # Apply speed reduction factor based on core index (linear scaling)
                speedup_factor = 1.0 - heterogeneity_factor * core_idx / max(1, num_cores - 1)
                # Calculate time, apply random variation, ensure minimum time
                time = max(1.0, base_time * speedup_factor * random.uniform(1 - variation_factor, 1 + variation_factor))
                core_times[task_id].append(time)

        # --- Calculate Cloud Execution Times (Send, Compute, Receive) ---
        # Relate these times to base local time, data size, and compute intensity.
        # Base factors for transfer and compute phases from the cloud range.
        base_transfer_factor = random.uniform(cloud_phases_range[0], cloud_phases_range[1])
        # Cloud compute might be relatively faster than transfers compared to local execution.
        base_compute_factor = random.uniform(cloud_phases_range[0] / 2, cloud_phases_range[1] / 2)

        # Send time depends mainly on data size.
        send_time = data_size * base_transfer_factor * random.uniform(0.8, 1.2)
        # Cloud compute time depends on base local time and compute intensity (higher intensity -> faster cloud relative to local).
        compute_time = (base_time / compute_intensity) * base_compute_factor * random.uniform(0.7, 1.3)
        # Receive time depends on data size (often smaller output data than input).
        receive_time = (data_size * 0.7) * base_transfer_factor * random.uniform(0.8, 1.2) # Assume output data is 70% of input

        # Store cloud phase times, ensuring minimum positive values.
        cloud_times[task_id] = [max(0.1, send_time), max(0.1, compute_time), max(0.1, receive_time)]

    # Return dictionaries containing the generated execution times.
    return core_times, cloud_times


def add_task_attributes(predefined_tasks,
                        num_edge_nodes, # Added parameter: needed for edge data sizes
                        data_size_range=(0.1, 1.0), # Range in MB
                        complexity_range=(0.5, 5.0), # Abstract complexity unit
                        data_intensity_range=(0.2, 2.0), # Abstract data intensity unit
                        task_type_weights=None, # Weights for 'compute', 'data', 'balanced'
                        scale_data_by_type=True, # Adjust data sizes based on task type?
                        seed=None):
    """
    Enhances a list of existing task objects by adding attributes like task type,
    complexity, data intensity, and detailed data transfer sizes (in MB) for all
    relevant links in a potential three-tier architecture (Device, Edge, Cloud).

    Args:
        predefined_tasks (list): List of Task objects (expected to have an 'id' attribute).
                                 Tasks are modified in-place.
        num_edge_nodes (int): Number of edge computing nodes available.
        data_size_range (tuple): (min, max) base data size in MB.
        complexity_range (tuple): (min, max) range for computational complexity factor.
        data_intensity_range (tuple): (min, max) range for data intensity factor.
        task_type_weights (dict, optional): Dictionary mapping task type (str) to probability weight.
                                            Defaults to {'compute': 0.3, 'data': 0.3, 'balanced': 0.4}.
        scale_data_by_type (bool): If True, scales generated data sizes based on the assigned task type
                                   (e.g., 'data' tasks have larger transfers).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: The list of enhanced Task objects (modified in-place).
    """
    # Set random seed for reproducibility.
    if seed is not None:
        random.seed(seed)

    # Use default task type weights if none are provided.
    if task_type_weights is None:
        task_type_weights = {'compute': 0.3, 'data': 0.3, 'balanced': 0.4}

    # Prepare for weighted random choice of task types.
    task_types = list(task_type_weights.keys())
    weights = list(task_type_weights.values())

    data_size_min, data_size_max = data_size_range

    # Process each task in the input list.
    for task in predefined_tasks:
        # Randomly assign a task type based on the specified weights.
        task.task_type = random.choices(task_types, weights=weights)[0]

        # Assign computational complexity biased by task type.
        if task.task_type == 'compute':
            # Higher complexity range for compute-heavy tasks.
            task.complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
        elif task.task_type == 'data':
            # Lower complexity range for data-heavy tasks.
            task.complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
        else:  # 'balanced'
            # Standard range for balanced tasks.
            task.complexity = random.uniform(*complexity_range)

        # Assign data intensity biased by task type.
        if task.task_type == 'data':
            # Higher data intensity for data-heavy tasks.
            task.data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        elif task.task_type == 'compute':
            # Lower data intensity for compute-heavy tasks.
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        else:  # 'balanced'
            # Standard range for balanced tasks.
            task.data_intensity = random.uniform(*data_intensity_range)

        # Initialize dictionary to hold data sizes for various transfers.
        task.data_sizes = {}

        # Determine data transfer size ranges based on task type, if scaling is enabled.
        if scale_data_by_type:
            if task.task_type == 'data':
                # Data tasks: Larger upload and download sizes.
                dev_cloud_up_range = (data_size_max * 0.6, data_size_max)
                cloud_dev_down_range = (data_size_max * 0.5, data_size_max * 0.9)
            elif task.task_type == 'compute':
                # Compute tasks: Smaller upload and download sizes.
                dev_cloud_up_range = (data_size_min, data_size_min + (data_size_max - data_size_min) * 0.3)
                cloud_dev_down_range = (data_size_min, data_size_min + (data_size_max - data_size_min) * 0.2)
            else:  # 'balanced'
                # Balanced tasks: Medium upload and download sizes.
                dev_cloud_up_range = (data_size_min + (data_size_max - data_size_min) * 0.3, data_size_min + (data_size_max - data_size_min) * 0.7)
                cloud_dev_down_range = (data_size_min + (data_size_max - data_size_min) * 0.2, data_size_min + (data_size_max - data_size_min) * 0.6)
        else:
            # If no scaling, use the same base range for all task types.
            dev_cloud_up_range = cloud_dev_down_range = data_size_range

        # --- Generate Data Sizes for All Potential Transfers ---

        # 1. Device <-> Cloud (These are the base values)
        ds_dev_cloud_up = random.uniform(*dev_cloud_up_range)     # Base upload size
        ds_cloud_dev_down = random.uniform(*cloud_dev_down_range) # Base download size
        task.data_sizes['device_to_cloud'] = ds_dev_cloud_up
        task.data_sizes['cloud_to_device'] = ds_cloud_dev_down

        # 2. Device <-> Edge (Derived from Device <-> Cloud base values)
        for i in range(1, num_edge_nodes + 1):
            # Device to Edge: Typically less data than full cloud upload (e.g., partial data, features).
            task.data_sizes[f'device_to_edge{i}'] = ds_dev_cloud_up * random.uniform(0.6, 0.9) # 60-90% of cloud upload
            # Edge to Device: Typically less data than full cloud download (e.g., processed results).
            task.data_sizes[f'edge{i}_to_device'] = ds_cloud_dev_down * random.uniform(0.7, 1.0) # 70-100% of cloud download

        # 3. Edge <-> Cloud (Derived from Device <-> Cloud base values)
        for i in range(1, num_edge_nodes + 1):
             # Edge to Cloud: Might be similar or slightly more than device->cloud if edge aggregates/relays data.
            task.data_sizes[f'edge{i}_to_cloud'] = ds_dev_cloud_up * random.uniform(0.8, 1.2) # 80-120% of base upload
            # Cloud to Edge: Might be similar or slightly more than cloud->device
            task.data_sizes[f'cloud_to_edge{i}'] = ds_cloud_dev_down * random.uniform(0.8, 1.2) # 80-120% of base download

        # 4. Edge <-> Edge (More complex - depends on workflow, derived here from Device <-> Edge)
        for i in range(1, num_edge_nodes + 1):
            for j in range(1, num_edge_nodes + 1):
                if i == j: continue # Skip transfer to self
                key = f'edge{i}_to_edge{j}'
                base_size = min(task.data_sizes.get(f'device_to_edge{i}', 0.01), # Use default if key missing
                                task.data_sizes.get(f'device_to_edge{j}', 0.01))
                task.data_sizes[key] = base_size * random.uniform(0.5, 1.1) # 50-110% of the minimum relevant input

        # Ensure all generated data sizes are positive (minimum 0.01 MB).
        for key in task.data_sizes:
            task.data_sizes[key] = max(0.01, task.data_sizes[key])

        # print(f"Task {getattr(task, 'id', 'N/A')}: Type={task.task_type}, Complexity={task.complexity:.2f}, Data Intensity={task.data_intensity:.2f}")
        # print(f"  Data sizes (MB): {task.data_sizes}")

    # Return the list of tasks (which have been modified in-place).
    return predefined_tasks

def generate_task_graph(num_tasks=40, complexity_level="medium", num_cores=3, num_edge_nodes=0, num_edge_cores=0, core_times=None, cloud_times=None, edge_times=None, seed=None, complexity_params=None):
    if seed is not None:
        random.seed(seed)
    
    # Map complexity level to connectivity parameters
    if complexity_params is None:
        complexity_params = {}
        
    if complexity_level == "low":
        connectivity = complexity_params.get('connectivity', 0.15)
        min_pred_ratio = complexity_params.get('min_pred_ratio', 0.1)
        max_pred_ratio = complexity_params.get('max_pred_ratio', 0.3)
    elif complexity_level == "medium":
        connectivity = complexity_params.get('connectivity', 0.3)
        min_pred_ratio = complexity_params.get('min_pred_ratio', 0.2)
        max_pred_ratio = complexity_params.get('max_pred_ratio', 0.5)
    else:  # "high"
        connectivity = complexity_params.get('connectivity', 0.5)
        min_pred_ratio = complexity_params.get('min_pred_ratio', 0.3)
        max_pred_ratio = complexity_params.get('max_pred_ratio', 0.7)
    
    # Helper function to create Task with execution times
    def _create_task(task_id, succ_list=None):
        from mcc_extended import Task  # Import here to avoid circular imports
        return Task(
            id=task_id,
            succ_task=succ_list or [],
            core_times=core_times,
            cloud_times=cloud_times,
            edge_times=edge_times,
            num_cores=num_cores,
            num_edge_nodes=num_edge_nodes,
            num_edge_cores=num_edge_cores
        )
    
    # Organize tasks into layers
    num_layers = max(3, min(num_tasks // 3, 5))
    layers = [[] for _ in range(num_layers)]
    remaining = num_tasks
    layer_sizes = []
    
    # First layer (entry tasks)
    first_layer_size = max(1, num_tasks // (2 * num_layers))
    layer_sizes.append(first_layer_size)
    remaining -= first_layer_size
    
    # Middle layers
    for i in range(1, num_layers - 1):
        size = max(1, int(remaining * random.uniform(0.2, 0.4)))
        size = min(size, remaining - 1)
        layer_sizes.append(size)
        remaining -= size
    
    # Last layer (exit tasks)
    layer_sizes.append(remaining)
    
    # Create task objects layer by layer
    next_id = 1
    tasks = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        layer_tasks = []
        for _ in range(layer_size):
            task = _create_task(next_id)
            layer_tasks.append(task)
            tasks.append(task)
            next_id += 1
        layers[layer_idx] = layer_tasks
    
    # Ensure minimum connectivity between adjacent layers
    for i in range(num_layers - 1):
        if layers[i] and layers[i + 1]:
            source = random.choice(layers[i])
            target = random.choice(layers[i + 1])
            if target not in source.succ_tasks:
                source.succ_tasks.append(target)
                target.pred_tasks.append(source)
    
    # Add controlled connectivity based on complexity parameters
    for layer_idx in range(1, num_layers):
        for task in layers[layer_idx]:
            all_possible_preds = [t for i in range(layer_idx) for t in layers[i]]
            min_connections = max(1, int(len(all_possible_preds) * min_pred_ratio * connectivity))
            max_connections = max(min_connections, int(len(all_possible_preds) * max_pred_ratio * connectivity))
            max_connections = min(max_connections, len(all_possible_preds))
            
            existing_preds = len(task.pred_tasks)
            needed_min = max(0, min_connections - existing_preds)
            needed_max = max(0, max_connections - existing_preds)
            
            if needed_max > 0:
                potential_preds = [t for t in all_possible_preds if t not in task.pred_tasks]
                if potential_preds:
                    num_to_add = random.randint(needed_min, needed_max) if needed_min < needed_max else needed_min
                    num_to_add = min(num_to_add, len(potential_preds))
                    if num_to_add > 0:
                        new_preds = random.sample(potential_preds, num_to_add)
                        for pred in new_preds:
                            task.pred_tasks.append(pred)
                            pred.succ_tasks.append(task)
    
    # Ensure proper entry tasks (no predecessors)
    for task in layers[0]:
        task.pred_tasks = []
    
    # Ensure at least one exit task
    exit_found = any(not task.succ_tasks for task in layers[-1])
    if not exit_found and layers[-1]:
        exit_task = random.choice(layers[-1])
        for task in tasks:
            if exit_task in task.succ_tasks:
                task.succ_tasks.remove(exit_task)
        exit_task.succ_tasks = []
    
    # Validate connectivity (ensure all tasks are reachable)
    def validate_connectivity():
        entry_tasks = [t for t in tasks if not t.pred_tasks]
        reachable = set()
        visited = set()
        stack = entry_tasks.copy()
        
        while stack:
            task = stack.pop()
            if task in visited:
                continue
            visited.add(task)
            reachable.add(task)
            for succ in task.succ_tasks:
                if succ not in visited:
                    stack.append(succ)
        
        if len(reachable) != len(tasks):
            unreachable = [t for t in tasks if t not in reachable]
            for task in unreachable:
                if entry_tasks:
                    entry_task = entry_tasks[0]
                    entry_task.succ_tasks.append(task)
                    task.pred_tasks.append(entry_task)
    
    validate_connectivity()
    
    # Validate acyclicity (ensure IDs are properly ordered)
    for task in tasks:
        for pred in task.pred_tasks:
            if pred.id >= task.id:
                raise ValueError(f"Invalid DAG: Task {pred.id} is a predecessor of Task {task.id}")
    
    return tasks

def generate_configs(param_ranges: dict = None, sampling_method: str = 'grid', num_samples: int = None, seed: int = 42):
    """
    Generates a list of MCCConfiguration objects, representing different simulation scenarios.
    It can create configurations based on parameter ranges (using grid or random sampling)
    and also includes a set of predefined, specialized configurations (e.g., favoring
    local execution, cloud execution, edge execution, or specific conditions like low battery).

    Args:
        param_ranges (dict, optional): Dictionary mapping parameter names (str) to their
                                       ranges or lists of values. Used for generating sweep
                                       configurations. Keys should match MCCConfiguration args.
                                       Example: {'num_cores': [2, 4], 'bandwidth_factor': (0.5, 1.5)}
                                       If None, only specialized configurations are returned.
        sampling_method (str): Method for sampling from `param_ranges`:
                               'grid': Creates configurations for all combinations (Cartesian product).
                               'random': Creates `num_samples` configurations with random values from ranges.
        num_samples (int, optional): Number of random samples to generate if `sampling_method` is 'random'.
        seed (int): Random seed for reproducibility of sampling and configuration generation.

    Returns:
        list: A list containing MCCConfiguration objects, representing the generated scenarios.
    """
    combined_configs = [] # List to hold all generated configurations

    # Set random seeds for reproducibility of sampling and generation processes.
    random.seed(seed)
    np.random.seed(seed)

    # --- Generate Sweep Configurations (if param_ranges is provided) ---
    if param_ranges is not None:
        # Define how many points to sample for continuous parameters in grid search.
        # None means use the range directly (for discrete int parameters).
        discretization = {
            'bandwidth_factor': 5,            # e.g., 5 points between min and max bandwidth factor.
            'power_factor': 5,
            'rf_efficiency': 4,
            'time_constraint_multiplier': 5,
            'battery_level': 3,               # e.g., High, Medium, Low battery levels.
            'num_cores': None,                # Discrete parameter; use provided range/list.
            'num_edge_nodes': None,           # Discrete parameter.
            'num_edge_cores': None,           # Discrete parameter.
        }

        # Validate the sampling method argument.
        if sampling_method not in ('grid', 'random'):
            raise ValueError("sampling_method must be 'grid' or 'random'")

        if sampling_method == 'grid':
            # --- Grid Sampling ---
            param_space = {} # Stores the discrete values to use for each parameter

            # Build the parameter space from the provided ranges.
            for param, range_val in param_ranges.items():
                # Handle different types of range specifications.
                if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
                     min_val, max_val = range_val # Standard (min, max) tuple/list
                elif isinstance(range_val, (int, float)):
                     min_val, max_val = range_val, range_val # Fixed value provided
                else:
                    # Assume it's already a list/tuple of specific values to use.
                     param_space[param] = list(range_val)
                     continue # Skip further processing for this parameter

                # Determine the values to use based on discretization.
                if param in discretization and discretization[param] is not None:
                    # Continuous parameter with specified number of points.
                    param_space[param] = np.linspace(min_val, max_val, discretization[param])
                elif isinstance(min_val, int) and isinstance(max_val, int):
                    # Discrete integer parameter (e.g., num_cores). Use all integers in range.
                    param_space[param] = list(range(min_val, max_val + 1))
                elif isinstance(min_val, float) or isinstance(max_val, float):
                    # Continuous parameter with default discretization (if not specified above).
                    param_space[param] = np.linspace(min_val, max_val, 5) # Default to 5 points
                else:
                     # Fallback for other types (e.g., categorical, already handled).
                     if param not in param_space:
                         param_space[param] = [min_val] if min_val == max_val else [min_val, max_val]

            # Get the Cartesian product of all parameter values to form all combinations.
            param_names = list(param_space.keys())
            param_values = [param_space[name] for name in param_names]

            # Create an MCCConfiguration object for each combination.
            for values in itertools.product(*param_values):
                params = dict(zip(param_names, values)) # Combine names and values

                # --- Sanity Checks/Adjustments for Edge ---
                # If no edge nodes, ensure no edge cores are specified.
                if params.get('num_edge_nodes', 0) == 0:
                    params['num_edge_cores'] = 0
                # If edge nodes exist but cores are 0, default to 1 core per node.
                elif params.get('num_edge_cores', 0) == 0 and params.get('num_edge_nodes', 0) > 0:
                     params['num_edge_cores'] = 1

                # Build a unique name for the configuration based on its parameters.
                name_parts = []
                for param, value in params.items():
                    # Create a short abbreviation for the parameter name.
                    abbrev = "".join(c for c in param if c.isupper()) or param[:3]
                    if isinstance(value, float):
                        name_parts.append(f"{abbrev}{value:.1f}") # Format floats
                    else:
                        name_parts.append(f"{abbrev}{value}") # Format other types
                config_name = "Sweep_" + "_".join(name_parts) # Prefix with "Sweep_"

                # Create and add the configuration object.
                config = MCCConfiguration(name=config_name, seed=seed, **params)
                combined_configs.append(config)

        elif sampling_method == 'random':
            # --- Random Sampling ---
            if num_samples is None:
                raise ValueError("num_samples must be provided for random sampling.")

            # Generate the specified number of random configurations.
            for i in range(num_samples):
                params = {}
                # Sample a value for each parameter from its specified range.
                for param, range_val in param_ranges.items():
                     # Handle different range types.
                     if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
                         min_val, max_val = range_val
                     elif isinstance(range_val, (int, float)):
                         min_val, max_val = range_val, range_val # Fixed value
                     else:
                          params[param] = random.choice(list(range_val)) # Choose from a list of specific values
                          continue

                     # Sample based on type (int, float, etc.).
                     if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                     elif isinstance(min_val, float) or isinstance(max_val, float):
                        params[param] = random.uniform(min_val, max_val)
                     else: # Fallback for other types (e.g., assume categorical).
                        params[param] = random.choice([min_val] if min_val == max_val else [min_val, max_val])

                # --- Sanity Checks/Adjustments for Edge ---
                if params.get('num_edge_nodes', 0) == 0:
                    params['num_edge_cores'] = 0
                elif params.get('num_edge_cores', 0) == 0 and params.get('num_edge_nodes', 0) > 0:
                     params['num_edge_cores'] = 1

                config_name = f"Random_Sample_{i + 1}"
                # Use incremental seeds for random configurations for better diversity if needed later.
                config = MCCConfiguration(name=config_name, seed=seed + i, **params)
                combined_configs.append(config)

    # --- Generate Specialized, Predefined Configurations ---
    specialized_configs = []

    # Define example efficient core profiles (low power).
    efficient_cores = [
        { 'idle_power': 0.01, 'power_coefficient': 0.2, 'power_base': 0.02, 'frequency_range': (0.5, 1.5), 'current_frequency': 1.2 },
        { 'idle_power': 0.03, 'power_coefficient': 0.5, 'power_base': 0.05, 'frequency_range': (0.6, 1.8), 'current_frequency': 1.5 },
        { 'idle_power': 0.05, 'power_coefficient': 1.0, 'power_base': 0.1, 'frequency_range': (0.8, 2.0), 'current_frequency': 1.8 }
    ]

    # Define example power-hungry core profiles (high performance, high power).
    power_hungry_cores = [
        { 'idle_power': 0.15, 'power_coefficient': 2.5, 'power_base': 0.3, 'frequency_range': (1.0, 3.0), 'current_frequency': 2.8 },
        { 'idle_power': 0.12, 'power_coefficient': 2.0, 'power_base': 0.25, 'frequency_range': (1.0, 2.8), 'current_frequency': 2.5 },
        { 'idle_power': 0.1, 'power_coefficient': 1.8, 'power_base': 0.2, 'frequency_range': (0.8, 2.5), 'current_frequency': 2.2 }
    ]

    # --- TWO-TIER Specialized Configurations (No Edge Nodes) ---
    # Configurations favoring local execution (efficient cores, slightly worse network/RF).
    for num_cores in [2, 3]:
        config = MCCConfiguration(
            name=f"Local-Favoring_Cores_{num_cores}", num_cores=num_cores,
            power_factor=0.5, rf_efficiency=0.7, bandwidth_factor=0.8,
            num_edge_nodes=0, num_edge_cores=0, # Explicitly two-tier
            seed=seed
        )
        # Assign the predefined efficient core profiles to this config.
        config.core_power_profiles = efficient_cores[:num_cores]
        specialized_configs.append(config)

    # Configurations favoring cloud execution (power-hungry cores, good network/RF).
    for bandwidth in [2.0, 3.0]: # High bandwidth factors
        config = MCCConfiguration(
            name=f"Cloud-Favoring_BW_{bandwidth:.1f}", num_cores=3,
            power_factor=1.5, rf_efficiency=1.3, bandwidth_factor=bandwidth,
            num_edge_nodes=0, num_edge_cores=0, # Explicitly two-tier
            seed=seed
        )
        # Assign the predefined power-hungry core profiles.
        config.core_power_profiles = power_hungry_cores
        specialized_configs.append(config)

    # Configuration simulating a critically low battery level.
    battery_config = MCCConfiguration(
        name="Battery-Critical_15pct", num_cores=3,
        battery_level=15.0, # Low battery level activates battery_factor in power models
        power_factor=1.0, rf_efficiency=1.0, bandwidth_factor=1.0, # Standard other factors
        num_edge_nodes=0, num_edge_cores=0, # Explicitly two-tier
        seed=seed
    )
    battery_config.core_power_profiles = efficient_cores[:3] # Assume efficient cores in low battery scenario
    specialized_configs.append(battery_config)

    # --- THREE-TIER Specialized Configurations (With Edge Nodes) ---
    # A baseline three-tier configuration with moderate parameters.
    specialized_configs.append(MCCConfiguration(
        name="Three-Tier_Base", num_cores=3, num_edge_nodes=2, num_edge_cores=2,
        bandwidth_factor=1.0, power_factor=1.0, rf_efficiency=1.0, battery_level=80.0,
        seed=seed
    ))

    # Configuration favoring edge execution (good network, relatively efficient edge/device).
    specialized_configs.append(MCCConfiguration(
        name="Edge-Favoring", num_cores=3, num_edge_nodes=3, num_edge_cores=2,
        bandwidth_factor=1.5, # Good network connectivity
        power_factor=0.8,     # Edge/Device slightly more power efficient overall
        rf_efficiency=1.1,    # Efficient RF
        seed=seed
    ))

     # Configuration simulating network constraints, particularly affecting edge.
     # Achieved here by a low global bandwidth factor.
    specialized_configs.append(MCCConfiguration(
        name="Network-Constrained_Edge", num_cores=3, num_edge_nodes=2, num_edge_cores=2,
        bandwidth_factor=0.5, # Slow network overall
        power_factor=1.0, rf_efficiency=1.0, # Standard power factors
        seed=seed
    ))

    # Configuration with highly heterogeneous edge nodes.
    # Heterogeneity is introduced within generate_mcc_power_models based on node/core IDs.
    specialized_configs.append(MCCConfiguration(
        name="Heterogeneous_Edge", num_cores=3, num_edge_nodes=4, num_edge_cores=2,
        bandwidth_factor=1.0, power_factor=1.0, rf_efficiency=1.0, # Standard factors
        seed=seed # Seed ensures consistent heterogeneity pattern across runs
    ))


    # Combine the configurations generated from sweeps and the specialized ones.
    combined_configs.extend(specialized_configs)

    # Ensure unique configuration names by appending a count if duplicates exist.
    names = set()
    final_configs = []
    for cfg in combined_configs:
        original_name = cfg.name
        count = 1
        # Check if name already exists, append suffix if needed.
        while cfg.name in names:
            cfg.name = f"{original_name}_{count}"
            count += 1
        names.add(cfg.name) # Add the unique name to the set
        final_configs.append(cfg)

    # Return the final list of unique configuration objects.
    return final_configs

DEFAULT_RANDOM_RANGES = {
    'num_cores': (2, 5),             # e.g., 2 to 5 cores
    'bandwidth_factor': (0.5, 2.5),  # Wide range for network quality
    'power_factor': (0.7, 1.3),    # Overall power efficiency variation
    'rf_efficiency': (0.8, 1.2),     # RF efficiency variation
    'battery_level': (20.0, 100.0),  # Battery level range
    'time_constraint_multiplier': (1.3, 2.5), # How much slack vs fastest time
    'num_edge_nodes': (0, 4),        # 0 to 4 edge nodes (includes two-tier)
    'num_edge_cores': (1, 4)         # 1 to 4 cores per edge node (if nodes > 0)
    # Add other relevant MCCConfiguration parameters here if needed
    # 'local_time_range': ((1, 10), (5, 20)), # Example if range itself needs random selection
    # 'data_size_range': ((0.1, 2.0), (1.0, 8.0))
}

def generate_single_random_config(
    name_prefix: str = "Random_Test",
    param_ranges: dict = None,
    base_seed: int = None
    ):
    """
    Generates a single MCCConfiguration object with parameters chosen randomly
    from specified ranges.

    Args:
        name_prefix (str): A prefix for the generated configuration name.
        param_ranges (dict, optional): Dictionary mapping parameter names to their
                                       ranges or lists of values. Uses
                                       DEFAULT_RANDOM_RANGES if None or for missing keys.
        base_seed (int, optional): A seed to initialize the random number generator.
                                    If None, randomness will not be reproducible.
                                    A unique seed is derived for the specific config.

    Returns:
        MCCConfiguration: A single randomly generated configuration object.
    """
    # Use provided ranges, falling back to defaults for missing parameters
    effective_ranges = DEFAULT_RANDOM_RANGES.copy()
    if param_ranges:
        effective_ranges.update(param_ranges)

    # Seed the generator if a base seed is provided
    # Use a temporary random state for this generation to avoid interfering
    # with global random state if called multiple times without reseeding.
    rand_state = np.random.RandomState(base_seed)
    # Derive a unique seed for this specific config generation based on the base_seed
    # This helps ensure different calls with the same base_seed don't produce
    # the exact same sequence of random numbers internally if generation logic has steps.
    config_specific_seed = rand_state.randint(0, 2**32 - 1) if base_seed is not None else None
    # Re-seed the temporary generator for this specific config
    if config_specific_seed is not None:
        rand_state.seed(config_specific_seed)


    params = {}
    # Sample a value for each parameter from its effective range
    for param, range_val in effective_ranges.items():
        # Handle different range types
        if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
            min_val, max_val = range_val
        elif isinstance(range_val, (int, float)):
            min_val, max_val = range_val, range_val # Fixed value
        else:
             # Assume it's a list of specific choices
             params[param] = rand_state.choice(list(range_val))
             continue

        # Sample based on type (int, float)
        if isinstance(min_val, int) and isinstance(max_val, int):
            params[param] = rand_state.randint(min_val, max_val + 1) # Use randint (inclusive)
        elif isinstance(min_val, float) or isinstance(max_val, float):
            params[param] = rand_state.uniform(min_val, max_val)
        else: # Fallback for less common types
            params[param] = rand_state.choice([min_val] if min_val == max_val else [min_val, max_val])

    # --- Sanity Checks/Adjustments for Edge ---
    # If no edge nodes selected, ensure no edge cores are specified.
    if params.get('num_edge_nodes', 0) == 0:
        params['num_edge_cores'] = 0
    # If edge nodes exist but cores somehow ended up 0, default to 1 core per node.
    elif params.get('num_edge_cores', 0) == 0 and params.get('num_edge_nodes', 0) > 0:
         params['num_edge_cores'] = 1 # Ensure at least 1 core if edge nodes exist

    # Generate a name for the config - can add more detail if needed
    # Include parts of the seed or a timestamp for uniqueness if base_seed is None
    timestamp_or_seed = f"s{base_seed}" if base_seed is not None else f"t{int(time.time())}"
    # Add a couple of key parameters to the name
    core_info = f"C{params.get('num_cores', 'N')}"
    edge_info = f"E{params.get('num_edge_nodes', 'N')}x{params.get('num_edge_cores', 'N')}"
    config_name = f"{name_prefix}_{core_info}_{edge_info}_{timestamp_or_seed}"

    # Create and return the configuration object
    # Pass the derived config_specific_seed to MCCConfiguration
    # Ensure MCCConfiguration accepts 'seed'
    return MCCConfiguration(name=config_name, seed=config_specific_seed, **params)

class MCCConfiguration:
    """
    Represents a complete configuration bundle for an MCC task scheduling simulation run.
    It centralizes all parameters needed to define a scenario, including device, network,
    task characteristics, and edge tier settings. Facilitates creating and managing
    diverse test scenarios easily.
    """
    def __init__(self,
                 name: str = "Default Config",       # Unique name for the configuration
                 num_tasks: int = 40,               # Number of tasks in the application DAG
                 num_cores: int = 3,                # Number of cores on the mobile device
                 bandwidth_factor: float = 1.0,     # Multiplier for network bandwidths
                 power_factor: float = 1.0,         # Multiplier for power consumption values
                 rf_efficiency: float = 1.0,        # Efficiency of RF components (lower = more power)
                 battery_level: float = 100.0,      # Mobile device battery level (%)
                 time_constraint_multiplier: float = 1.5, # Factor applied to a baseline time to set T_max deadline
                 task_type_distribution: dict = None, # Weights for task types {'compute':w1, 'data':w2, ..}
                 local_time_range: tuple = (1, 15),   # Base execution time range on fastest local core
                 cloud_phases_range: tuple = (0.5, 3.0),# Base time factor range for cloud transfers
                 data_size_range: tuple = (0.1, 5.0),   # Base data size range (MB)
                 num_edge_nodes: int = 0,           # Number of available edge nodes (0 disables edge)
                 num_edge_cores: int = 0,           # Number of cores per edge node
                 seed: int = None):                 # Random seed for reproducibility
        """
        Initializes the MCCConfiguration object.
        """
        self.name = name
        self.num_tasks = num_tasks
        self.num_cores = num_cores
        self.bandwidth_factor = bandwidth_factor
        self.power_factor = power_factor
        self.rf_efficiency = rf_efficiency
        self.battery_level = battery_level
        self.time_constraint_multiplier = time_constraint_multiplier

        # Use default task distribution if none is provided to avoid mutable default arg issues.
        self.task_type_distribution = task_type_distribution if task_type_distribution is not None else {
            'compute': 0.3, 'data': 0.3, 'balanced': 0.4
        }
        self.local_time_range = local_time_range
        self.cloud_phases_range = cloud_phases_range
        self.data_size_range = data_size_range

        # Ensure consistency: if no edge nodes, then no edge cores should be considered.
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores = num_edge_cores if num_edge_nodes > 0 else 0

        self.seed = seed
        # Placeholder for specific core power profiles (can be set by specialized generators).
        self.core_power_profiles = None

    def apply(self):
        """
        Applies the configuration settings to generate all necessary simulation components
        (power models, network conditions, base task execution times) based on the
        parameters stored in this configuration object.

        Note: Edge execution time generation depends on task objects which are typically
              created after calling apply(), so it's handled separately in the main simulation loop.

        Returns:
            dict: A dictionary containing the generated simulation inputs:
                  - Configuration parameters needed downstream (e.g., num_tasks, num_cores).
                  - Generated models (power_models, upload_rates, download_rates).
                  - Generated base data (core_execution_times, cloud_execution_times).
                  - Parameters needed for subsequent generation steps (e.g., data_size_range).
        """
        # Set random seed for reproducibility across all generation steps within this apply call.
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed) # Also seed numpy if used internally by generation functions

        # 1. Generate power models (device, edge, cloud, RF).
        #    Includes generating 'nodeFactor' for edge heterogeneity.
        power_models = generate_mcc_power_models(
            device_type='mobile', # Assume primary device perspective
            battery_level=self.battery_level,
            power_factor=self.power_factor,
            rf_efficiency=self.rf_efficiency,
            num_cores=self.num_cores,
            num_edge_nodes=self.num_edge_nodes,
            num_edge_cores=self.num_edge_cores,
            core_power_profiles=self.core_power_profiles, # Use specific profiles if set
            seed=self.seed
        )

        # 2. Generate base task execution times for device cores and cloud phases.
        #    Updates the global dictionaries (consider refactoring to return values instead).
        global core_execution_times, cloud_execution_times
        core_times, cloud_times = generate_task_execution_times(
            self.num_tasks,
            self.num_cores,
            self.local_time_range,
            self.cloud_phases_range,
            seed=self.seed
        )
        # Update global state - necessary for edge time generation later if using globals.
        core_execution_times = core_times
        cloud_execution_times = cloud_times

        # 3. Generate network conditions (bandwidths for all links).
        upload_rates, download_rates = generate_mcc_network_conditions(
            self.bandwidth_factor,
            seed=self.seed
        )

        # 4. Initialize edge execution times dictionary.
        #    Actual calculation using generate_edge_task_execution_times is deferred
        #    until Task objects are created in the main simulation script, as it needs
        #    access to task attributes and previously generated times (via globals or passed data).
        global edge_execution_times
        edge_execution_times = {} # Initialize as empty, will be populated later by generate_edge_task_execution_times

        # Return a dictionary containing all generated components and key parameters.
        return {
            # Configuration parameters needed by the main simulation/scheduler
            'num_tasks': self.num_tasks,
            'num_cores': self.num_cores,
            'num_edge_nodes': self.num_edge_nodes,
            'num_edge_cores': self.num_edge_cores,
            'time_constraint_multiplier': self.time_constraint_multiplier,
            'seed': self.seed,
            # Generated models and data structures
            'core_execution_times': core_times,         # {task_id: [core0_t, core1_t, ...]}
            'cloud_execution_times': cloud_times,       # {task_id: [send_t, compute_t, receive_t]}
            'edge_execution_times': edge_execution_times, # Initially empty, populated later {task_id: {(eid,cid):t}}
            'upload_rates': upload_rates,             # {link_type: rate_mbps}
            'download_rates': download_rates,           # {link_type: rate_mbps}
            'power_models': power_models,             # Nested dict of power models
            # Parameters needed for subsequent generation (e.g., adding task attributes)
            'data_size_range': self.data_size_range,
            'task_type_distribution': self.task_type_distribution,
        }

    def __str__(self):
        """Return a human-readable string representation of the configuration for logging."""
        # Determine architecture type based on edge node presence.
        arch_type = "Three-Tier" if self.num_edge_nodes > 0 else "Two-Tier"
        # Format edge information string.
        edge_info = f", Edge: {self.num_edge_nodes} nodes x {self.num_edge_cores} cores/node" if self.num_edge_nodes > 0 else ""
        # Construct the full representation string.
        return (
            f"Config Name: {self.name}\n"
            f"  Architecture: {arch_type} (Device Cores: {self.num_cores}{edge_info})\n"
            f"  Tasks: {self.num_tasks}\n"
            f"  Network: BW Factor={self.bandwidth_factor:.2f}\n"
            f"  Power: Factor={self.power_factor:.2f}, RF Eff={self.rf_efficiency:.2f}, Battery={self.battery_level:.1f}%\n"
            f"  Constraints: Time Mult={self.time_constraint_multiplier:.2f}\n"
            f"  Data Sizes (MB): {self.data_size_range}\n"
            f"  Seed: {self.seed}"
        )


