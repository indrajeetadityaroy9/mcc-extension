#mcc_extended_q_learning.py

from copy import deepcopy
import bisect
from collections import deque, defaultdict
import numpy as np
from heapq import heappush, heappop
import os
import time
import pandas as pd
from dataclasses import dataclass, field
import random
import pickle
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime

from data import (
    ExecutionTier, edge_execution_times,
    generate_edge_task_execution_times,generate_task_graph,
    generate_configs, MCCConfiguration, generate_single_random_config
)

from mcc_extended import (
    Task, total_time, total_energy, execution_unit_selection,
    construct_sequence, kernel_algorithm, apply_configuration_parameters, assign_task_attributes
)

from validation import validate_task_dependencies

class QState:
    """
    Enhanced state representation that includes task distribution and characteristics
    across different execution tiers to provide more context for learning.
    """
    def __init__(self, tasks: List[Task], T_current: float, E_current: float):
        # Basic state information
        self.assignments = [task.assignment for task in tasks]
        self.T_current = T_current
        self.E_current = E_current
        
        # Calculate tier distribution
        self.tier_distribution = {}
        for task in tasks:
            tier = task.execution_tier
            self.tier_distribution[tier] = self.tier_distribution.get(tier, 0) + 1
        
        # Calculate tier characteristics (e.g., average complexity and data intensity)
        self.tier_characteristics = self._calculate_tier_characteristics(tasks)
    
    def _extract_assignments(self, tasks: List[Task]) -> List[int]:
        """Extract a normalized assignment vector from tasks."""
        return [task.assignment for task in tasks]
    
    def _calculate_tier_distribution(self, tasks: List[Task]) -> Dict[ExecutionTier, int]:
        """Calculate how many tasks are assigned to each tier."""
        distribution = {tier: 0 for tier in ExecutionTier}
        for task in tasks:
            if task.execution_tier in distribution:
                distribution[task.execution_tier] += 1
        return distribution
    
    def _calculate_tier_characteristics(self, tasks: List[Task]) -> Dict:
        """Calculate average complexity and data intensity for each tier."""
        tier_data = {tier: {'count': 0, 'complexity_sum': 0, 'data_intensity_sum': 0} 
                     for tier in ExecutionTier}
        
        # Sum up values by tier
        for task in tasks:
            tier = task.execution_tier
            if tier in tier_data:
                tier_data[tier]['count'] += 1
                tier_data[tier]['complexity_sum'] += getattr(task, 'complexity', 1.0)
                tier_data[tier]['data_intensity_sum'] += getattr(task, 'data_intensity', 1.0)
        
        # Calculate averages and discretize to reduce state space
        result = {}
        for tier, data in tier_data.items():
            if data['count'] > 0:
                avg_complexity = data['complexity_sum'] / data['count']
                avg_data_intensity = data['data_intensity_sum'] / data['count']
                # Discretize to reduce state space (5 bins)
                result[tier] = {
                    'complexity': round(avg_complexity * 5) / 5,
                    'data_intensity': round(avg_data_intensity * 5) / 5
                }
            else:
                result[tier] = {'complexity': 0, 'data_intensity': 0}
        
        return result
    
    def _compute_hash(self) -> int:
        """Compute a hash that uniquely identifies this state."""
        # Discretize continuous values to prevent too many unique states
        discretized_time = round(self.T_current, 1)  # One decimal place
        discretized_energy = round(self.E_current, 1)  # One decimal place
        
        # Include tier distribution and key characteristics in hash
        tier_info = tuple(sorted((tier.value, count) for tier, count in self.tier_distribution.items()))
        
        # Create a tuple of all components for hashing
        state_tuple = (
            tuple(self.assignments),
            discretized_time,
            discretized_energy,
            tier_info,
            # Add discretized characteristics
            tuple(sorted((tier.value, data['complexity'], data['data_intensity']) 
                         for tier, data in self.tier_characteristics.items()))
        )
        
        return hash(state_tuple)
    
    def __hash__(self):
        """Allow QState objects to be hashable."""
        return self._compute_hash()
    
    def __eq__(self, other) -> bool:
        """Compare two states for equality."""
        if not isinstance(other, QState):
            return False
        
        return (self.assignments == other.assignments and
                abs(self.T_current - other.T_current) < 0.11 and 
                abs(self.E_current - other.E_current) < 0.11 and
                self.tier_distribution == other.tier_distribution and
                self.tier_characteristics == other.tier_characteristics)
    
    def __str__(self):
        """String representation for debugging."""
        dist_str = ", ".join([f"{tier.name}:{count}" for tier, count in self.tier_distribution.items()])
        return f"EnhancedQState(T={self.T_current:.2f}, E={self.E_current:.2f}, Dist=[{dist_str}])"

class QAction:
    """
    Represents an action in the Q-learning framework.
    An action is a task migration decision: which task to move and where.
    """
    def __init__(self, task_idx: int, target_unit_idx: int):
        """
        Initialize an action (task migration).
        
        Args:
            task_idx: Index of the task to migrate (0-based)
            target_unit_idx: Index of the target execution unit (0-based)
        """
        self.task_idx = task_idx
        self.target_unit_idx = target_unit_idx
    
    def __hash__(self) -> int:
        """Hash for dictionary lookups."""
        return hash((self.task_idx, self.target_unit_idx))
    
    def __eq__(self, other) -> bool:
        """Compare two actions for equality."""
        if not isinstance(other, QAction):
            return False
        return (self.task_idx == other.task_idx and
                self.target_unit_idx == other.target_unit_idx)
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"QAction(task={self.task_idx+1}, target={self.target_unit_idx})"


class QLearningOptimizer:
    """
    Enhanced Q-Learning optimizer for Mobile Cloud Computing task scheduling.
    Improvements include:
    - Less aggressive early stopping
    - Adaptive exploration rate
    - Learning rate scheduling
    - Improved reward function
    - Experience replay buffer
    """
    def __init__(self, 
                 tasks: List[Task],
                 sequence: List[List[int]],
                 T_max: float,
                 device_power_profiles: Dict,
                 rf_power: Dict,
                 upload_rates: Dict,
                 download_rates: Dict,
                 num_cores: int,
                 num_edge_nodes: int,
                 num_edge_cores: int,
                 alpha: float = 0.5,         # Learning rate
                 gamma: float = 0.9,         # Discount factor
                 epsilon_start: float = 1.0, # Initial exploration probability
                 epsilon_end: float = 0.1,   # Final exploration probability
                 epsilon_decay: float = 0.9, # Rate of exploration decay
                 time_penalty_factor: float = 100.0,  # Penalty factor for time violations
                 energy_reward_factor: float = 10.0,  # Reward factor for energy savings
                 max_episodes: int = 50,      # Number of learning episodes
                 max_iterations: int = 30,    # Maximum iterations per episode
                 replay_buffer_size: int = 100, # Size of experience replay buffer
                 alpha_min: float = 0.1,      # Minimum learning rate
                 alpha_decay: float = 0.99):  # Decay factor for learning rate
        """
        Initialize the Q-Learning optimizer with adaptive exploration and experience replay.
        """
        # Store reference to the task scheduling state
        self.tasks = tasks
        self.sequence = sequence
        self.T_max = T_max
        
        # Store the power model and network parameters
        self.device_power_profiles = device_power_profiles
        self.rf_power = rf_power
        self.upload_rates = upload_rates
        self.download_rates = download_rates
        
        # System configuration
        self.num_cores = num_cores
        self.num_edge_nodes = num_edge_nodes
        self.num_edge_cores = num_edge_cores
        self.num_edge_units = num_edge_nodes * num_edge_cores
        self.total_units = num_cores + 1 + self.num_edge_units
        
        # Q-learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Learning rate scheduling parameters (newly added)
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        
        # Algorithm control parameters
        self.max_episodes = max_episodes
        self.max_iterations = max_iterations
        
        # Reward function parameters
        self.time_penalty_factor = time_penalty_factor
        self.energy_reward_factor = energy_reward_factor
        
        # Experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        
        # Initialize Q-table as a defaultdict for sparse representation
        self.q_table = defaultdict(float)
        
        # Cache to avoid redundant evaluations
        self.migration_cache = {}
        
        # Tracking for convergence and adaptation
        self.rewards_history = []
        self.recent_rewards = []  # For adaptive exploration
        self.energy_history = []
        
        # Current state parameters
        self.current_time = total_time(tasks)
        self.current_energy = total_energy(tasks, device_power_profiles, rf_power, upload_rates)
        
        # Tracking for migration statistics
        self.applied_migrations = []
        
        # Initialize valid actions (migration possibilities)
        self.migration_choices = self._initialize_migration_choices()
        self.tier_metrics = {tier: {'task_count': 0, 
                                   'avg_complexity': 0, 
                                   'avg_data_intensity': 0} 
                            for tier in ExecutionTier}
        
    def _initialize_migration_choices(self) -> np.ndarray:
        """Initialize possible migration choices for each task."""
        # Create matrix of [num_tasks × total_execution_units]
        migration_choices = np.zeros((len(self.tasks), self.total_units), dtype=bool)
        
        # Set migration possibilities based on current assignments
        for i, task in enumerate(self.tasks):
            current_unit = task.assignment
            
            # Can't migrate to current location
            if 0 <= current_unit < self.total_units:
                migration_choices[i, :] = True
                migration_choices[i, current_unit] = False
            else:
                # If unassigned, all locations are potential targets
                migration_choices[i, :] = True
        
        return migration_choices
    
    def get_current_state(self) -> QState:
        """Get the current state representation for Q-learning."""
        return QState(self.tasks, self.current_time, self.current_energy)
    
    def get_valid_actions(self) -> List[QAction]:
        """Get all valid actions (task migrations) from the current state."""
        valid_actions = []
        
        # Build list from migration possibilities matrix
        for task_idx in range(len(self.tasks)):
            for target_unit_idx in range(self.total_units):
                if self.migration_choices[task_idx, target_unit_idx]:
                    valid_actions.append(QAction(task_idx, target_unit_idx))
        
        return valid_actions
    
    def select_action(self, state: QState, force_greedy: bool = False) -> Optional[QAction]:
        """
        Select an action using epsilon-greedy strategy with adaptive exploration.
        
        Args:
            state: Current state representation
            force_greedy: If True, always choose best action (no exploration)
            
        Returns:
            Selected action or None if no valid actions
        """
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return None
        
        # With probability epsilon, choose a random action (exploration)
        if random.random() < self.epsilon and not force_greedy:
            return random.choice(valid_actions)
        
        # Otherwise, choose the action with highest Q-value (exploitation)
        best_action = None
        best_q_value = float('-inf')
        
        for action in valid_actions:
            q_value = self.q_table[(state, action)]
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        # If all Q-values are zero, choose randomly
        if best_action is None:
            return random.choice(valid_actions)
        
        return best_action
    
    def evaluate_migration(self, task_idx: int, target_unit_idx: int) -> Tuple[float, float]:
        """Evaluate potential task migration outcome."""
        # Check cache first to avoid redundant calculations
        cache_key = (task_idx, target_unit_idx, tuple(task.assignment for task in self.tasks))
        if cache_key in self.migration_cache:
            return self.migration_cache[cache_key]
        
        # Create copies for evaluation
        tasks_copy = deepcopy(self.tasks)
        sequence_copy = [seq.copy() for seq in self.sequence]
        
        # Apply migration to the copies
        sequence_copy = construct_sequence(
            tasks_copy,
            task_idx + 1,  # Task ID is 1-based
            target_unit_idx,
            sequence_copy,
            self.num_cores,
            self.num_edge_nodes,
            self.num_edge_cores,
            self.upload_rates
        )
        
        # Reschedule with linear-time kernel algorithm
        tasks_copy = kernel_algorithm(
            tasks_copy,
            sequence_copy,
            self.num_cores,
            self.num_edge_nodes,
            self.num_edge_cores,
            self.upload_rates,
            self.download_rates
        )
        
        # Calculate new metrics
        new_time = total_time(tasks_copy)
        new_energy = total_energy(tasks_copy, self.device_power_profiles, self.rf_power, self.upload_rates)
        
        # Cache the result
        self.migration_cache[cache_key] = (new_time, new_energy)
        return new_time, new_energy
    
    def calculate_reward(self, time_before: float, energy_before: float, 
                     time_after: float, energy_after: float) -> float:
        """
        Enhanced reward function that balances energy reduction with time considerations.
        Implements non-linear scaling for energy rewards and adds components for time improvements.
        """
        # === Energy Reward Component (Primary Objective) ===
        energy_reduction = energy_before - energy_after
        
        # Handle the sign of energy change first to avoid complex numbers
        if energy_reduction >= 0:  # Energy decreased (good)
            # Apply non-linear scaling that rewards larger drops more
            if energy_before > 0.01:  # Avoid division by zero
                # Relative improvement with progressive scaling
                energy_reduction_ratio = energy_reduction / energy_before
                # Use a sigmoid-like function for smooth non-linear scaling
                energy_reward = (2.0 / (1.0 + math.exp(-8.0 * energy_reduction_ratio)) - 1.0) 
                energy_reward *= energy_before * self.energy_reward_factor
            else:
                # Fallback for very small baseline energy
                energy_reward = energy_reduction * self.energy_reward_factor
        else:  # Energy increased (bad)
            # Stronger penalty for energy increases
            energy_reward = 1.5 * energy_reduction * self.energy_reward_factor
        
        # === Time Component (Secondary Objective / Constraint) ===
        # Hard constraint violation penalty (if exceeding T_max)
        time_violation = max(0, time_after - self.T_max)
        # Quadratic penalty ensures stronger response as violation grows
        time_penalty = time_violation * time_violation * self.time_penalty_factor
        
        # Soft time management rewards/penalties
        time_diff = time_before - time_after  # Positive if time decreased
        
        # 1. Reward time reductions (even when within constraint)
        time_reduction_reward = 0
        if time_diff > 0:  # Time decreased
            # Scale reward by how close we already are to T_max
            proximity_to_deadline = max(0, 1.0 - (time_before / self.T_max))
            # Higher reward when close to deadline, smaller when far from it
            time_reduction_reward = time_diff * 5.0 * (0.5 + proximity_to_deadline)
        
        # 2. Penalty for time increases (even when within constraint)
        time_increase_penalty = 0
        if time_diff < 0:  # Time increased
            # Stronger penalty as we get closer to T_max
            proximity_to_deadline = max(0, time_after / self.T_max)
            time_increase_penalty = abs(time_diff) * 3.0 * proximity_to_deadline
        
        # 3. Balance reward for staying well under time constraint
        # Small reward for maintaining slack, diminishes as we approach deadline
        time_slack = max(0, self.T_max - time_after)
        time_slack_reward = min(2.0, time_slack * 0.05)
        
        # === Composite Reward ===
        # Main components with relative weights:
        # - Energy reward (main objective)
        # - Hard time constraint penalty (must satisfy)
        # - Soft time management components (secondary consideration)
        reward = (
            energy_reward -            # Primary energy objective
            time_penalty -             # Hard constraint (dominant when violated)
            time_increase_penalty +    # Soft penalty for time increases
            time_reduction_reward +    # Reward for time reductions
            time_slack_reward          # Small reward for deadline margin
        )
        
        return reward
    
    def update_q_value(self, state, action, next_state, reward):
        """
        Update Q-value using the Q-learning update rule with current learning rate.
        Modified to handle EnhancedQState objects safely.
        """
        # Convert states to hashable keys for dictionary lookup
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Get the current Q-value for this state-action pair
        current_q = self.q_table.get((state_key, action), 0.0)
        
        # Get maximum Q-value for the next state (over all actions)
        max_next_q = 0.0
        valid_next_actions = self.get_valid_actions()
        if valid_next_actions:
            next_q_values = [self.q_table.get((next_state_key, next_action), 0.0) 
                            for next_action in valid_next_actions]
            if next_q_values:
                max_next_q = max(next_q_values)
        
        # Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[(state_key, action)] = new_q
    
    def _get_state_key(self, state):
        """
        Convert a state object to a hashable key for the Q-table.
        This provides a safe way to use EnhancedQState objects as dictionary keys.
        """
        if hasattr(state, 'assignments') and hasattr(state, 'T_current') and hasattr(state, 'E_current'):
            # Create a tuple that captures the essential state information
            assignments_tuple = tuple(state.assignments)
            time_discrete = round(state.T_current, 1)
            energy_discrete = round(state.E_current, 1)
            
            # Include tier distribution if available
            tier_info = None
            if hasattr(state, 'tier_distribution'):
                tier_info = tuple(sorted((tier.value, count) 
                                        for tier, count in state.tier_distribution.items()))
            
            # Return a composite key
            return (assignments_tuple, time_discrete, energy_discrete, tier_info)
        else:
            # Fallback for simple states
            return str(state)
    
    def apply_migration(self, action: QAction) -> Tuple[float, float]:
        """Apply the selected migration to the actual tasks and sequences."""
        task_idx, target_unit_idx = action.task_idx, action.target_unit_idx
        
        # Record the migration details for tracking
        from_unit = self.tasks[task_idx].assignment
        
        # Track tier changes (for analysis)
        from_tier = self.tasks[task_idx].execution_tier
        
        # Apply the migration using construct_sequence
        self.sequence = construct_sequence(
            self.tasks,
            task_idx + 1,  # Task ID is 1-based
            target_unit_idx,
            self.sequence,
            self.num_cores,
            self.num_edge_nodes,
            self.num_edge_cores,
            self.upload_rates
        )
        
        # Reschedule with kernel algorithm
        self.tasks = kernel_algorithm(
            self.tasks,
            self.sequence,
            self.num_cores,
            self.num_edge_nodes,
            self.num_edge_cores,
            self.upload_rates,
            self.download_rates
        )
        
        # Update to new state
        new_time = total_time(self.tasks)
        new_energy = total_energy(self.tasks, self.device_power_profiles, self.rf_power, self.upload_rates)
        
        # Determine target tier based on execution unit
        if target_unit_idx < self.num_cores:
            to_tier = ExecutionTier.DEVICE
        elif target_unit_idx == self.num_cores:
            to_tier = ExecutionTier.CLOUD
        else:
            to_tier = ExecutionTier.EDGE
            
        # Record the migration
        self.applied_migrations.append({
            'task_id': task_idx + 1,  # 1-based task ID
            'from_unit': from_unit,
            'to_unit': target_unit_idx,
            'from_tier': from_tier.name if from_tier else 'Unknown',
            'to_tier': to_tier.name if to_tier else 'Unknown',
            'time_before': self.current_time,
            'time_after': new_time,
            'energy_before': self.current_energy,
            'energy_after': new_energy
        })
        
        # Update current state tracking
        self.current_time = new_time
        self.current_energy = new_energy
        
        # Update migration possibilities after this migration
        self.migration_choices = self._initialize_migration_choices()
        
        return new_time, new_energy
    
    def update_exploration_rate(self):
        """
        Adaptive exploration rate based on recent reward trends.
        """
        # Base decay
        self.epsilon = max(0.01, self.epsilon * 0.95)

        # Increase exploration if stuck in local minimum
        if len(self.recent_rewards) >= 5 and max(self.recent_rewards[-5:]) - min(self.recent_rewards[-5:]) < 0.1:
            self.epsilon = min(0.5, self.epsilon * 1.5)  # Boost exploration
            print(f"Boosting exploration rate to {self.epsilon:.2f} due to stagnation")

        # Reduce exploration after significant improvements
        if len(self.recent_rewards) >= 2 and self.recent_rewards[-1] > self.recent_rewards[-2] * 1.5:
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.7)  # Reduce exploration
            print(f"Reducing exploration rate to {self.epsilon:.2f} to exploit good solutions")
    
    def update_learning_rate(self):
        """
        Decay learning rate over time to allow more stable convergence.
        """
        # Simple decay with minimum threshold
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
    
    def add_to_replay_buffer(self, state, action, next_state, reward):
        """
        Add experience to replay buffer, maintaining size limit.
        """
        # Add the experience to the buffer
        self.replay_buffer.append((state, action, next_state, reward))
        
        # Keep buffer size limited
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)  # Remove oldest experience
    
    def replay_experiences(self, num_samples=5):
        """
        Replay experiences from buffer to improve learning.
        """
        if not self.replay_buffer:
            return
        
        # Sample experiences from buffer (with higher probability for recent ones)
        buffer_size = len(self.replay_buffer)
        weights = [1.0 + i/buffer_size for i in range(buffer_size)]  # Higher weights for recent experiences
        
        # Sample with replacement, weighted by recency
        samples = random.choices(self.replay_buffer, weights=weights, k=min(num_samples, buffer_size))
        
        # Update Q-values from sampled experiences
        for state, action, next_state, reward in samples:
            self.update_q_value(state, action, next_state, reward)
    
    def optimize(self, max_episodes: int = 50, max_iterations_per_episode: int = 30, verbose: bool = True) -> Tuple[List[Task], List[List[int]], List[Dict]]:
        """
        Run the enhanced Q-learning optimization process with improved state representation and reward shaping.
        """
        if verbose:
            print(f"Starting enhanced Q-learning optimization with T_max = {self.T_max:.2f}")
            print(f"Initial state: T = {self.current_time:.2f}, E = {self.current_energy:.2f}")
        
        # Initialize best solution tracking
        best_energy = self.current_energy
        best_time = self.current_time  # Track best time as well
        best_tasks = deepcopy(self.tasks)
        best_sequence = [seq.copy() for seq in self.sequence]
        best_migrations = []
        
        # Initialize convergence tracking
        iterations_without_improvement = 0
        episodes_without_improvement = 0
        
        # Progress tracking
        self.rewards_history = []
        self.energy_history = [self.current_energy]
        self.time_history = [self.current_time]
        self.recent_rewards = []
        
        # Global iteration counter (across all episodes)
        global_iteration = 0
        
        # Run learning episodes
        for episode in range(max_episodes):
            if verbose and (episode % 10 == 0 or episode == max_episodes - 1):
                print(f"\nStarting episode {episode+1}/{max_episodes}, ε = {self.epsilon:.4f}, α = {self.alpha:.4f}")
            
            # Reset to best known state for this episode
            episode_tasks = deepcopy(best_tasks)
            episode_sequence = [seq.copy() for seq in best_sequence]
            
            # Reset state tracking for this episode
            episode_time = total_time(episode_tasks)
            episode_energy = total_energy(episode_tasks, self.device_power_profiles, self.rf_power, self.upload_rates)
            
            # Analyze task distribution by tier for enhanced exploration
            tier_distribution = {tier: 0 for tier in ExecutionTier}
            for task in episode_tasks:
                if hasattr(task, 'execution_tier') and task.execution_tier in tier_distribution:
                    tier_distribution[task.execution_tier] += 1
            
            # Run one episode
            episode_iterations_without_improvement = 0
            for iteration in range(max_iterations_per_episode):
                global_iteration += 1
                
                # Create temporary optimizer for this episode
                temp_optimizer = QLearningOptimizer(
                    tasks=episode_tasks,
                    sequence=episode_sequence,
                    T_max=self.T_max,
                    device_power_profiles=self.device_power_profiles,
                    rf_power=self.rf_power,
                    upload_rates=self.upload_rates,
                    download_rates=self.download_rates,
                    num_cores=self.num_cores,
                    num_edge_nodes=self.num_edge_nodes,
                    num_edge_cores=self.num_edge_cores,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    epsilon_start=self.epsilon,
                    epsilon_end=self.epsilon_end,
                    epsilon_decay=self.epsilon_decay,
                    time_penalty_factor=self.time_penalty_factor,
                    energy_reward_factor=self.energy_reward_factor,
                    max_episodes=self.max_episodes,
                    max_iterations=self.max_iterations,
                    replay_buffer_size=self.replay_buffer_size,
                    alpha_min=self.alpha_min,
                    alpha_decay=self.alpha_decay
                )
                
                # Share the Q-table, cache, and replay buffer with the temporary optimizer
                temp_optimizer.q_table = self.q_table
                temp_optimizer.migration_cache = self.migration_cache
                temp_optimizer.replay_buffer = self.replay_buffer
                
                # Get current state using enhanced state representation
                state = temp_optimizer.get_current_state()  # Now returns EnhancedQState
                
                # Select action using epsilon-greedy
                action = temp_optimizer.select_action(state)
                if action is None:
                    # No valid actions remaining
                    break
                
                # Evaluate the migration outcome
                time_before = episode_time
                energy_before = episode_energy
                
                # Simulate the migration to get the next state
                time_after, energy_after = temp_optimizer.evaluate_migration(
                    action.task_idx, action.target_unit_idx
                )
                
                # Calculate reward with improved reward function
                reward = temp_optimizer.calculate_reward(
                    time_before, energy_before, time_after, energy_after
                )
                
                # Create next state representation using enhanced state
                next_state = QState(episode_tasks, time_after, energy_after)
                
                # Update Q-value
                temp_optimizer.update_q_value(state, action, next_state, reward)
                
                # Add to replay buffer
                temp_optimizer.add_to_replay_buffer(state, action, next_state, reward)
                
                # Replay some experiences to reinforce learning
                temp_optimizer.replay_experiences(num_samples=4)  # Increased from 3 to 4
                
                # Apply the migration to episode state
                episode_time, episode_energy = temp_optimizer.apply_migration(action)
                
                # Update episode tracking
                episode_tasks = temp_optimizer.tasks
                episode_sequence = temp_optimizer.sequence
                
                # Track reward for convergence analysis
                self.rewards_history.append(reward)
                self.recent_rewards.append(reward)
                self.energy_history.append(episode_energy)
                self.time_history.append(episode_time)
                
                # Define improvement criteria (weighted combination of energy and time)
                # The main objective is still energy reduction while staying under time constraint
                improvement_detected = False
                
                # Check if time constraint is satisfied
                time_constraint_satisfied = episode_time <= self.T_max
                
                if time_constraint_satisfied:
                    # First priority: If energy is better and time constraint is met
                    if episode_energy < best_energy:
                        improvement_detected = True
                        improvement_message = f"Better energy: E = {episode_energy:.2f} (prev: {best_energy:.2f}), T = {episode_time:.2f}"
                    
                    # Second priority: If energy is the same but time is better
                    elif abs(episode_energy - best_energy) < 0.01 and episode_time < best_time:
                        improvement_detected = True
                        improvement_message = f"Same energy with better time: E = {episode_energy:.2f}, T = {episode_time:.2f} (prev: {best_time:.2f})"
                
                # Special case: If we've never found a valid solution, accept first one that meets time constraint
                if not time_constraint_satisfied and best_energy == self.current_energy and episode_energy < best_energy:
                    improvement_detected = True
                    improvement_message = f"First valid solution: E = {episode_energy:.2f}, T = {episode_time:.2f}"
                
                # Update best solution if improvement detected and time constraint is satisfied
                if improvement_detected and time_constraint_satisfied:
                    episode_iterations_without_improvement = 0
                    iterations_without_improvement = 0
                    episodes_without_improvement = 0
                    
                    if verbose:
                        print(f"  Iteration {iteration+1}: Found {improvement_message}")
                    
                    best_energy = episode_energy
                    best_time = episode_time
                    best_tasks = deepcopy(episode_tasks)
                    best_sequence = [seq.copy() for seq in episode_sequence]
                    best_migrations = temp_optimizer.applied_migrations
                else:
                    episode_iterations_without_improvement += 1
                    iterations_without_improvement += 1
                
                # Check for episode convergence (stuck in local optimum)
                if episode_iterations_without_improvement >= max_iterations_per_episode // 2:
                    # If we're stuck for half the iterations, break and try a new episode
                    break
                    
                # Check for global convergence only after a minimum number of episodes
                if episode > 5:
                    # Adaptive exploration boost based on progress
                    if iterations_without_improvement > 100:
                        # If we're seeing no progress after many iterations
                        if self.epsilon < 0.4:
                            self.epsilon = 0.8  # More aggressive boost
                            iterations_without_improvement = 0
                            if verbose:
                                print(f"  Boosting exploration to {self.epsilon:.2f} for renewed search")
                    
                    # Early stopping if progress stagnates for too long
                    if episodes_without_improvement >= 20 and episode > max_episodes // 2:
                        if verbose:
                            print(f"  Early stopping after {episode+1} episodes - no improvement for 20 episodes")
                        break
            
            # Update episodes without improvement counter
            if iterations_without_improvement > 0:
                episodes_without_improvement += 1
            
            # Update exploration and learning rates between episodes
            self.update_exploration_rate()
            self.update_learning_rate()
            
            # Report progress
            if verbose and (episode % 10 == 0 or episode == max_episodes - 1):
                print(f"  Episode {episode+1} completed: Current best E = {best_energy:.2f}, T = {best_time:.2f}")
                tier_dist = {tier.name: sum(1 for task in best_tasks if task.execution_tier == tier) 
                            for tier in ExecutionTier}
                print(f"  Tasks by tier: {tier_dist}")
                print(f"  Q-table size: {len(self.q_table)} entries, Cache size: {len(self.migration_cache)} entries")
        
        # Final update to optimizer state
        self.tasks = best_tasks
        self.sequence = best_sequence
        self.applied_migrations = best_migrations
        self.current_time = best_time
        self.current_energy = best_energy
        
        if verbose:
            print(f"\nOptimization completed: Final E = {self.current_energy:.2f}, T = {self.current_time:.2f}")
            print(f"Total migrations: {len(self.applied_migrations)}")
            tier_dist = {tier.name: sum(1 for task in best_tasks if task.execution_tier == tier) 
                    for tier in ExecutionTier}
            print(f"Final task distribution by tier: {tier_dist}")
        
        return best_tasks, best_sequence, self.applied_migrations

def optimize_schedule_q_learning(
    tasks, sequence, T_max,
    device_power_profiles, rf_power, upload_rates, download_rates,
    num_cores, num_edge_nodes, num_edge_cores,
    alpha=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9,
    time_penalty_factor=100.0, energy_reward_factor=10.0,
    max_episodes=50, max_iterations=30, 
    replay_buffer_size=100, verbose=True, alpha_min=0.1, alpha_decay=0.99):
    """
    Implements Q-learning based task migration algorithm.
    Optimizes energy consumption while maintaining completion time constraints.
    
    Args:
        # ... [other args]
        replay_buffer_size: Size of experience replay buffer
        verbose: Whether to print progress information
        
    Returns:
        tuple: (tasks, sequence, migrations) with optimized scheduling
    """
    # Initialize Q-learning optimizer
    optimizer = QLearningOptimizer(
        tasks=tasks,
        sequence=sequence,
        T_max=T_max,
        device_power_profiles=device_power_profiles,
        rf_power=rf_power,
        upload_rates=upload_rates,
        download_rates=download_rates,
        num_cores=num_cores,
        num_edge_nodes=num_edge_nodes, 
        num_edge_cores=num_edge_cores,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        time_penalty_factor=time_penalty_factor,
        energy_reward_factor=energy_reward_factor,
        max_episodes=max_episodes,
        max_iterations=max_iterations,
        replay_buffer_size=replay_buffer_size,
        alpha_min = alpha_min,
        alpha_decay = alpha_decay
    )
    
    # Run optimization
    optimized_tasks, final_sequence, migration_history = optimizer.optimize(
        max_episodes=max_episodes,
        max_iterations_per_episode=max_iterations,
        verbose=verbose
    )
    
    return optimized_tasks, final_sequence, migration_history


def run_unified_test_q_learning(config: MCCConfiguration, q_learning_params=None):
    """
    Run a full test (initial schedule + Q-learning optimization) for a given configuration,
    supporting both two-tier and three-tier architectures.

    Args:
        config: MCCConfiguration object.
        q_learning_params: Dictionary with Q-learning hyperparameters (optional)

    Returns:
        dict: Test results and metrics, including initial and final task states.
    """
    start_run_time = time.time()
    print("-" * 60)
    print(f"Running Q-Learning Test: {config.name}")
    print(config) # Print config details

    # 1. Generate simulation parameters based on config
    params = apply_configuration_parameters(config)

    # 2. Create task graph structure
    tasks = generate_task_graph(
        num_tasks=40,  # Default value
        complexity_level="high",  # Default to high complexity 
        num_cores=config.num_cores,
        num_edge_nodes=config.num_edge_nodes,
        num_edge_cores=config.num_edge_cores,
        core_times=params['core_execution_times'],
        cloud_times=params['cloud_execution_times'],
        edge_times={},
        seed=config.seed  # Use config seed for reproducibility
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

    # --- Energy Optimization (Q-Learning Task Migration) ---
    T_max = T_initial * config.time_constraint_multiplier
    print(f" Optimizing schedule for energy using Q-Learning (T_max = {T_max:.2f})...")
    optimize_start_time = time.time()

    # *** Use the *initial scheduled state* as the starting point for optimization ***
    tasks_to_optimize = deepcopy(tasks_initial_state)
    sequence_to_optimize = [seq.copy() for seq in sequence_initial]

    # Define default Q-learning parameters
    default_q_params = {
        'alpha': 0.5,             # Learning rate
        'gamma': 0.9,             # Discount factor
        'epsilon_start': 1.0,     # Initial exploration probability
        'epsilon_end': 0.1,       # Final exploration probability
        'epsilon_decay': 0.9,     # Rate of exploration decay
        'time_penalty_factor': 100.0,  # Penalty for time violations
        'energy_reward_factor': 10.0,  # Reward for energy savings
        'max_episodes': 50,       # Number of learning episodes
        'max_iterations': 30      # Max iterations per episode
    }
    
    # Merge with user-provided parameters
    if q_learning_params:
        q_params = {**default_q_params, **q_learning_params}
    else:
        q_params = default_q_params

    # Run Q-learning optimization
    tasks_final, sequence_final, migrations = optimize_schedule_q_learning(
        tasks_to_optimize, sequence_to_optimize, T_max,
        device_power_profiles, rf_power, upload_rates, download_rates,
        config.num_cores, config.num_edge_nodes, config.num_edge_cores,
        alpha=q_params['alpha'],
        gamma=q_params['gamma'],
        epsilon_start=q_params['epsilon_start'],
        epsilon_end=q_params['epsilon_end'],
        epsilon_decay=q_params['epsilon_decay'],
        time_penalty_factor=q_params['time_penalty_factor'],
        energy_reward_factor=q_params['energy_reward_factor'],
        max_episodes=q_params['max_episodes'],
        max_iterations=q_params['max_iterations']
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
         },
        'q_learning_params': q_params  # Include the Q-learning parameters used
    }
    
    # Add distribution counts to the results
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

    return result_data


# Function to run comparison between heuristic and Q-learning methods
def run_comparison_test(config: MCCConfiguration):
    """
    Run a comparison test between the heuristic approach and Q-learning approach.
    
    Args:
        config: MCCConfiguration object
        
    Returns:
        dict: Comparison results
    """
    from mcc_extended import run_unified_test_3_tier
    
    print("-" * 60)
    print(f"Running Comparison Test: {config.name}")
    print(config)
    
    # Q-Learning parameters for the comparison
    q_params = {
        'alpha': 0.5,             # Learning rate
        'gamma': 0.9,             # Discount factor
        'epsilon_start': 1.0,     # Initial exploration probability
        'epsilon_end': 0.1,       # Final exploration probability
        'epsilon_decay': 0.9,     # Rate of exploration decay
        'time_penalty_factor': 100.0,  # Penalty for time violations
        'energy_reward_factor': 10.0,  # Reward for energy savings
        'max_episodes': 30,       # Number of learning episodes
        'max_iterations': 20      # Max iterations per episode
    }
    
    # Run heuristic approach
    print("\n--- Running Heuristic Approach ---")
    heuristic_result = run_unified_test_3_tier(config)
    
    # Run Q-learning approach
    print("\n--- Running Q-Learning Approach ---")
    q_learning_result = run_unified_test_q_learning(config, q_learning_params=q_params)
    
    # Compare results
    print("\n=== Comparison Results ===")
    print(f"Initial State:  T = {heuristic_result.get('initial_time', 0):.2f}, E = {heuristic_result.get('initial_energy', 0):.2f}")
    print(f"Time Constraint: T_max = {heuristic_result.get('time_constraint', 0):.2f}")
    print(f"Heuristic Final: T = {heuristic_result.get('final_time', 0):.2f}, E = {heuristic_result.get('final_energy', 0):.2f}, "
          f"Reductions: {heuristic_result.get('energy_reduction_percent', 0):.2f}%, Migrations: {heuristic_result.get('migration_count', 0)}")
    print(f"Q-Learning Final: T = {q_learning_result.get('final_time', 0):.2f}, E = {q_learning_result.get('final_energy', 0):.2f}, "
          f"Reductions: {q_learning_result.get('energy_reduction_percent', 0):.2f}%, Migrations: {q_learning_result.get('migration_count', 0)}")
    
    # Improvement analysis
    energy_diff = q_learning_result.get('final_energy', 0) - heuristic_result.get('final_energy', 0)
    energy_diff_percent = (energy_diff / heuristic_result.get('final_energy', 1)) * 100
    
    if energy_diff < 0:
        print(f"Q-Learning achieved {abs(energy_diff_percent):.2f}% BETTER energy efficiency than the heuristic approach.")
    elif energy_diff > 0:
        print(f"Heuristic was {energy_diff_percent:.2f}% more energy efficient than Q-Learning.")
    else:
        print("Both approaches achieved identical energy efficiency.")
        
    # Check time constraint satisfaction
    h_time_violation = max(0, heuristic_result.get('final_time', 0) - heuristic_result.get('time_constraint', float('inf')))
    q_time_violation = max(0, q_learning_result.get('final_time', 0) - q_learning_result.get('time_constraint', float('inf')))
    
    if h_time_violation > 1e-6:
        print(f"WARNING: Heuristic approach violated time constraint by {h_time_violation:.2f} units.")
    if q_time_violation > 1e-6:
        print(f"WARNING: Q-Learning approach violated time constraint by {q_time_violation:.2f} units.")
    
    # Return comparison results
    return {
        'config_name': config.name,
        'initial_time': heuristic_result.get('initial_time', 0),
        'initial_energy': heuristic_result.get('initial_energy', 0),
        'time_constraint': heuristic_result.get('time_constraint', 0),
        'heuristic_final_time': heuristic_result.get('final_time', 0),
        'heuristic_final_energy': heuristic_result.get('final_energy', 0),
        'heuristic_energy_reduction': heuristic_result.get('energy_reduction_percent', 0),
        'heuristic_migration_count': heuristic_result.get('migration_count', 0),
        'q_learning_final_time': q_learning_result.get('final_time', 0),
        'q_learning_final_energy': q_learning_result.get('final_energy', 0),
        'q_learning_energy_reduction': q_learning_result.get('energy_reduction_percent', 0),
        'q_learning_migration_count': q_learning_result.get('migration_count', 0),
        'q_learning_improvement': -energy_diff_percent,
        'heuristic_time_violation': h_time_violation,
        'q_learning_time_violation': q_time_violation,
        'heuristic_result': heuristic_result,
        'q_learning_result': q_learning_result
    }

if __name__ == "__main__":
    import traceback
    from datetime import datetime
    
    # --- Control Flags ---
    run_specialized = True      # Run the predefined specific scenarios
    run_random = True           # Run randomly generated scenarios
    run_complex = True          # Run tests with complex task graphs
    run_comparison = True       # Run direct comparison between Q-learning and heuristic
    print_schedules = True      # Print detailed schedule comparison after each run
    num_random_tests = 2        # How many random configurations to generate and test
    num_complex_tests = 2       # How many complex task graph tests to run
    save_results = True         # Save results to CSV
    
    # --- Q-Learning Parameters ---
    q_learning_params = {
        'alpha': 0.5,             # Learning rate
        'gamma': 0.9,             # Discount factor
        'epsilon_start': 1.0,     # Initial exploration probability
        'epsilon_end': 0.1,       # Final exploration probability
        'epsilon_decay': 0.9,     # Rate of exploration decay
        'time_penalty_factor': 100.0,  # Penalty for time violations
        'energy_reward_factor': 10.0,  # Reward for energy savings
        'max_episodes': 30,       # Number of learning episodes
        'max_iterations': 20      # Max iterations per episode
    }
    
    # --- Lists to store results ---
    all_results = []
    all_failures = []
    configs_run_count = 0
    
    # Time stamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"qmcc_results_{timestamp}"
    if save_results:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not create output directory: {e}")
            save_results = False
    
    # --- Section 1: Run Specific Specialized Tests ---
    if run_specialized:
        target_config_names = [
            "Local-Favoring_Cores_3",
            "Cloud-Favoring_BW_2.0",
            "Battery-Critical_15pct",
            "Three-Tier_Base",
            "Edge-Favoring",
            "Network-Constrained_Edge",
            "Heterogeneous_Edge"
        ]
        print("-" * 20 + " Running SPECIFIC Specialized Configurations (Q-Learning) " + "-" * 20)
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
            print(f"\n--- Starting execution of {len(configs_to_test_specialized)} specialized tests with Q-Learning ---")
            
            for i, config in enumerate(configs_to_test_specialized):
                test_name = f"Specialized Test {i+1}/{len(configs_to_test_specialized)}"
                config_id_name = config.name
                print(f"\n--- {test_name}: Running '{config_id_name}' ---")
                configs_run_count += 1
                start_single_test = time.time()
                result = None
                
                try:
                    # --- Run the core test function with Q-learning ---
                    result = run_unified_test_q_learning(config=config, q_learning_params=q_learning_params)
                    
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
        print("\n" + "-" * 20 + " Running RANDOMLY Generated Configurations (Q-Learning) " + "-" * 20)
        
        base_random_seed = 123 # Use a base seed for the sequence
        
        print(f"--- Starting execution of {num_random_tests} random tests with Q-Learning ---")
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
                    name_prefix="RandomTest_QL",
                    base_seed=random_config_seed
                )
                config_id_name = random_config.name
                print(f"Generated Config: {config_id_name}")
                
                # --- Run the core test function with Q-learning ---
                result = run_unified_test_q_learning(config=random_config, q_learning_params=q_learning_params)
                
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
    
    # --- Section 3: Run Complex Task Graph Tests ---
    if run_complex:
        print("\n" + "-" * 20 + " Running Complex Task Graph Tests (Q-Learning) " + "-" * 20)
        
        # Define complex test scenarios
        complex_scenarios = [
            {"name": "Diamond_High", "num_tasks": 30, "complexity_level": "high", "pattern": "diamond"},
            {"name": "Layered_Medium", "num_tasks": 30, "complexity_level": "medium", "pattern": "layered"},
            {"name": "Irregular_High", "num_tasks": 30, "complexity_level": "high", "pattern": "irregular"},
            {"name": "Mixed_Medium", "num_tasks": 30, "complexity_level": "medium", "pattern": "mixed"}
        ]
        
        # Select scenarios to run
        scenarios_to_run = complex_scenarios[:num_complex_tests]
        
        print(f"--- Starting execution of {len(scenarios_to_run)} complex task graph tests ---")
        
        # Base configuration for all complex tests
        base_complex_config = generate_single_random_config(
            name_prefix="ComplexGraph",
            base_seed=456
        )
        
        for i, scenario in enumerate(scenarios_to_run):
            test_name = f"Complex Graph Test {i+1}/{len(scenarios_to_run)}"
            scenario_name = scenario["name"]
            config_id_name = f"{base_complex_config.name}_{scenario_name}"
            print(f"\n--- {test_name}: Running '{config_id_name}' ---")
            configs_run_count += 1
            start_single_test = time.time()
            result = None
            
            try:
                # Apply configuration parameters
                params = apply_configuration_parameters(base_complex_config)
                
                # Create the complex task graph
                tasks = generate_task_graph(
                    num_tasks=scenario["num_tasks"],
                    complexity_level=scenario["complexity_level"],
                    num_cores=base_complex_config.num_cores,
                    num_edge_nodes=base_complex_config.num_edge_nodes,
                    num_edge_cores=base_complex_config.num_edge_cores,
                    core_times=params['core_execution_times'],
                    cloud_times=params['cloud_execution_times'],
                    edge_times={},
                    seed=456 + i
                )
                
                # Assign attributes to tasks
                tasks = assign_task_attributes(tasks, base_complex_config)
                
                # Generate edge execution times if needed
                if base_complex_config.num_edge_nodes > 0 and base_complex_config.num_edge_cores > 0:
                    print(" Generating edge execution times...")
                    generate_edge_task_execution_times(
                        tasks=tasks,
                        mcc_edge_power_models=params['power_models'].get('edge', {}),
                        num_edge_nodes=base_complex_config.num_edge_nodes,
                        num_edge_cores=base_complex_config.num_edge_cores,
                        seed=base_complex_config.seed
                    )
                    edge_execution_times = {t.id: t.edge_execution_times for t in tasks if hasattr(t, 'edge_execution_times')}
                    params['edge_execution_times'] = edge_execution_times
                
                # Extract parameters
                upload_rates = params['upload_rates']
                download_rates = params['download_rates']
                power_models = params['power_models']
                device_power_profiles = power_models.get('device', {})
                rf_power = power_models.get('rf', {})
                
                # Perform initial scheduling
                print(" Performing initial scheduling (minimal delay)...")
                initial_start_time = time.time()
                tasks_for_initial_scheduling = deepcopy(tasks)
                sequence_initial = execution_unit_selection(
                    tasks_for_initial_scheduling,
                    num_cores=base_complex_config.num_cores,
                    num_edge_nodes=base_complex_config.num_edge_nodes,
                    num_edge_cores=base_complex_config.num_edge_cores,
                    upload_rates=upload_rates,
                    download_rates=download_rates
                )
                initial_schedule_time = time.time() - initial_start_time
                
                # Store initial state
                tasks_initial_state = deepcopy(tasks_for_initial_scheduling)
                
                # Calculate initial metrics
                T_initial = total_time(tasks_initial_state)
                E_initial = total_energy(tasks_initial_state, device_power_profiles, rf_power, upload_rates)
                print(f" Initial Schedule: T={T_initial:.2f}, E={E_initial:.2f} (took {initial_schedule_time:.2f}s)")
                
                # Set time constraint
                T_max = T_initial * base_complex_config.time_constraint_multiplier
                
                # Perform Q-learning optimization
                print(f" Optimizing schedule using Q-Learning (T_max = {T_max:.2f})...")
                optimize_start_time = time.time()
                
                tasks_to_optimize = deepcopy(tasks_initial_state)
                sequence_to_optimize = [seq.copy() for seq in sequence_initial]
                
                # Run Q-learning optimization
                tasks_final, sequence_final, migrations = optimize_schedule_q_learning(
                    tasks_to_optimize, sequence_to_optimize, T_max,
                    device_power_profiles, rf_power, upload_rates, download_rates,
                    base_complex_config.num_cores, base_complex_config.num_edge_nodes, 
                    base_complex_config.num_edge_cores,
                    alpha=q_learning_params['alpha'],
                    gamma=q_learning_params['gamma'],
                    epsilon_start=q_learning_params['epsilon_start'],
                    epsilon_end=q_learning_params['epsilon_end'],
                    epsilon_decay=q_learning_params['epsilon_decay'],
                    time_penalty_factor=q_learning_params['time_penalty_factor'],
                    energy_reward_factor=q_learning_params['energy_reward_factor'],
                    max_episodes=q_learning_params['max_episodes'],
                    max_iterations=q_learning_params['max_iterations']
                )
                
                optimize_schedule_time = time.time() - optimize_start_time
                
                # Calculate final metrics
                T_final = total_time(tasks_final)
                E_final = total_energy(tasks_final, device_power_profiles, rf_power, upload_rates)
                print(f" Optimized Schedule: T={T_final:.2f}, E={E_final:.2f} (took {optimize_schedule_time:.2f}s)")
                
                # Create result dictionary
                result = {
                    'config_name': config_id_name,
                    'num_tasks': scenario["num_tasks"],
                    'complexity_level': scenario["complexity_level"],
                    'pattern': scenario["pattern"],
                    'num_cores': base_complex_config.num_cores,
                    'num_edge_nodes': base_complex_config.num_edge_nodes,
                    'num_edge_cores': base_complex_config.num_edge_cores,
                    'initial_time': T_initial,
                    'final_time': T_final,
                    'time_constraint': T_max,
                    'initial_energy': E_initial,
                    'final_energy': E_final,
                    'time_change_percent': (T_final - T_initial) / T_initial * 100 if T_initial > 0 else 0,
                    'energy_reduction_percent': (E_initial - E_final) / E_initial * 100 if E_initial > 0 else 0,
                    'migration_count': len(migrations),
                    'initial_scheduling_duration': initial_schedule_time,
                    'optimization_duration': optimize_schedule_time,
                    'total_duration': initial_schedule_time + optimize_schedule_time,
                    'q_learning_params': q_learning_params
                }
                
                # Add result to list
                all_results.append(result)
                end_single_test = time.time()
                print(f"--- Test '{config_id_name}' Completed Successfully in {end_single_test - start_single_test:.2f}s ---")
                
            except Exception as e:
                end_single_test = time.time()
                error_msg = f"Unhandled Exception: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"!!! Test '{config_id_name}' Failed with Unhandled Exception after {end_single_test - start_single_test:.2f}s: {type(e).__name__} !!!")
                all_failures.append({'config_name': config_id_name, 'error': error_msg})
    
    # --- Section 4: Run Direct Comparison Tests ---
    if run_comparison:
        print("\n" + "-" * 20 + " Running Direct Comparison Tests (Q-Learning vs Heuristic) " + "-" * 20)
        
        # Initialize list to store comparison results
        comparison_results = []
        
        # Select configurations for comparison
        # Use 1 specialized and 1 random configuration
        comparison_configs = []
        
        if configs_to_test_specialized and len(configs_to_test_specialized) > 0:
            # Select the first specialized config
            comparison_configs.append(("Specialized", configs_to_test_specialized[0]))
        
        # Create a random configuration for comparison
        random_compare_config = generate_single_random_config(
            name_prefix="CompareTest",
            base_seed=789
        )
        comparison_configs.append(("Random", random_compare_config))
        
        # Run each comparison
        for config_type, config in comparison_configs:
            print(f"\n--- Running Comparison Test: {config_type} '{config.name}' ---")
            
            try:
                # Run comparison test
                comparison_result = run_comparison_test(config)
                
                # Store result
                comparison_results.append(comparison_result)
                
                # Print summary of comparison
                energy_diff_percent = comparison_result.get('q_learning_improvement', 0)
                if energy_diff_percent > 0:
                    print(f"Q-Learning was {energy_diff_percent:.2f}% more energy efficient than heuristic.")
                elif energy_diff_percent < 0:
                    print(f"Heuristic was {abs(energy_diff_percent):.2f}% more energy efficient than Q-Learning.")
                else:
                    print("Both approaches achieved identical energy efficiency.")
                
            except Exception as e:
                error_msg = f"Unhandled Exception: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"!!! Comparison Test for '{config.name}' Failed: {type(e).__name__} !!!")
                all_failures.append({'config_name': f"Comparison_{config.name}", 'error': error_msg})
    
    # --- Save Results ---
    if save_results and all_results:
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Save results CSV
            results_csv_path = os.path.join(output_dir, "qmcc_results.csv")
            results_df.to_csv(results_csv_path, index=False)
            print(f"\nResults saved to {results_csv_path}")
            
            # Save failures if any
            if all_failures:
                failures_df = pd.DataFrame(all_failures)
                failures_csv_path = os.path.join(output_dir, "qmcc_failures.csv")
                failures_df.to_csv(failures_csv_path, index=False)
                print(f"Failures saved to {failures_csv_path}")
            
            # Save comparison results if any
            if run_comparison and comparison_results:
                # Extract key comparison metrics
                comparison_data = []
                for result in comparison_results:
                    comparison_data.append({
                        'config_name': result.get('config_name'),
                        'initial_energy': result.get('initial_energy'),
                        'heuristic_final_energy': result.get('heuristic_final_energy'),
                        'q_learning_final_energy': result.get('q_learning_final_energy'),
                        'heuristic_energy_reduction': result.get('heuristic_energy_reduction'),
                        'q_learning_energy_reduction': result.get('q_learning_energy_reduction'),
                        'q_learning_improvement': result.get('q_learning_improvement'),
                        'heuristic_time_violation': result.get('heuristic_time_violation'),
                        'q_learning_time_violation': result.get('q_learning_time_violation')
                    })
                
                # Save comparison results
                comparison_df = pd.DataFrame(comparison_data)
                comparison_csv_path = os.path.join(output_dir, "qmcc_comparison.csv")
                comparison_df.to_csv(comparison_csv_path, index=False)
                print(f"Comparison results saved to {comparison_csv_path}")
                
        except Exception as e:
            print(f"\nError saving results: {e}")
    
    # --- Combined Summary of ALL Results ---
    print("\n" + "=" * 35 + " Overall Q-Learning Test Run Summary " + "=" * 35)
    print(f"Total configurations attempted/run: {configs_run_count}")
    print(f"Successful tests: {len(all_results)}")
    print(f"Failed tests: {len(all_failures)}")
    
    if all_results:
        print("\nKey Metrics from Successful Runs:")
        print("-" * 105)
        print(f"{'Config Name':<30} | {'T_init':>8} | {'T_final':>8} | {'T_max':>8} | {'E_init':>10} | {'E_final':>10} | {'Migr':>4} | {'Reduc%':>6}")
        print("-" * 105)
        for r in all_results:
            if r and 'config_name' in r and 'error' not in r:
                reduc_perc = r.get('energy_reduction_percent', 0)
                t_max_val = r.get('time_constraint', 0)
                migr_count = r.get('migration_count', 'N/A')
                print(f"{r.get('config_name', 'N/A'):<30} | {r.get('initial_time', 0):>8.2f} | {r.get('final_time', 0):>8.2f} | {t_max_val:>8.2f} | "
                      f"{r.get('initial_energy', 0):>10.2f} | {r.get('final_energy', 0):>10.2f} | {migr_count:>4} | {reduc_perc:>6.1f}")
            else:
                 print(f"{'Malformed/Failed Result':<30} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>4} | {'N/A':>6}")
        print("-" * 105)
        
        # Check for deadline violations
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
    
    # Display comparison summary if run
    if run_comparison and comparison_results:
        print("\nComparison Results (Q-Learning vs. Heuristic):")
        print("-" * 105)
        print(f"{'Config':<20} | {'Initial E':>10} | {'Heuristic E':>10} | {'Q-Learning E':>12} | {'H Reduc%':>8} | {'QL Reduc%':>9} | {'QL Improv%':>9}")
        print("-" * 105)
        for result in comparison_results:
            config_name = result.get('config_name', 'N/A')
            initial_energy = result.get('initial_energy', 0)
            h_energy = result.get('heuristic_final_energy', 0)
            ql_energy = result.get('q_learning_final_energy', 0)
            h_reduc = result.get('heuristic_energy_reduction', 0)
            ql_reduc = result.get('q_learning_energy_reduction', 0)
            ql_improv = result.get('q_learning_improvement', 0)
            
            print(f"{config_name[:20]:<20} | {initial_energy:>10.2f} | {h_energy:>10.2f} | {ql_energy:>12.2f} | {h_reduc:>8.2f} | {ql_reduc:>9.2f} | {ql_improv:>9.2f}")
        print("-" * 105)
    
    if all_failures:
        print("\nFailures Encountered:")
        for f in all_failures:
            print(f" - Config: {f.get('config_name', 'Unknown Config')}")
            print(f"   Error: {f.get('error', 'Unknown Error').splitlines()[0]}")
            
    print("\n--- End of Overall Q-Learning Test Run ---")
