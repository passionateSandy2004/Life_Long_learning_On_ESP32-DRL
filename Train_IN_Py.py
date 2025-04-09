import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
import socket
import time
import threading
import sys

# Create a custom GameEnv class that wraps the existing implementation
class GameEnvironment:
    def __init__(self, render=False):
        # Import here to avoid import issues
        sys.path.append('.')
        from RL_env import GameEnv
        
        # Initialize pygame first
        pygame.init()
        
        # Create the actual environment
        self.env = GameEnv(render=render)
        self.screen_width = 800  # Match the screen width in RL_env.py
        self.render = render
        
    def set_render(self, render):
        """Allow toggling rendering at runtime"""
        if self.render == render:
            return  # No change needed
            
        self.render = render
        
        # Recreate environment with new render setting
        from RL_env import GameEnv
        self.env = GameEnv(render=render)
        
        if render:
            print("\nRendering enabled - now showing gameplay")
            print_game_controls()
        else:
            print("\nRendering disabled - training at maximum speed")
        
    def reset(self):
        # Process events to keep the window responsive
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        return self.env.reset()
    
    def step(self, action):
        # Process events to keep the window responsive
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        
        try:
            # Call the step method and catch any errors
            next_state, reward, done = self.env.step(action)
            
            # Ensure the display is updated
            if self.render:
                pygame.display.flip()
                
            return next_state, reward, done
            
        except AttributeError as e:
            # If 'render_screen' is missing, we'll handle the error
            if "render_screen" in str(e):
                # Get the next state without rendering
                next_state = self.env.get_state()
                
                # Handle rendering manually
                if self.render:
                    self.env.screen.fill((0, 0, 0))  # Black background
                    self.env.all_sprites.draw(self.env.screen)
                    pygame.display.flip()
                
                # Return a default reward and done=False
                return next_state, 0.0, False
            else:
                # Re-raise if it's a different attribute error
                raise e

# Network dimensions - must match the Arduino code
INPUT_DIM = 5
HIDDEN_DIM = 16
OUTPUT_DIM = 3

# Training hyperparameters - matching Arduino values
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.999
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
DISCOUNT_FACTOR = 0.95

# Experience replay settings
MEMORY_SIZE = 500
BATCH_SIZE = 32
MIN_MEMORY_SIZE = 32
TARGET_UPDATE_FREQ = 50

# Neural network architecture - matches Arduino implementation
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        return self.fc3(x2)

# Experience replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.main_network = DQN()
        self.target_network = DQN()
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=INITIAL_LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.epsilon = INITIAL_EPSILON
        self.learning_rate = INITIAL_LEARNING_RATE
        self.steps = 0
        
        # Success tracking (as in Arduino code)
        self.success_window = 20
        self.success_history = [0] * self.success_window
        self.success_index = 0
        self.success_rate = 0.0
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def select_action(self, state, is_bullet_active=False):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Calculate dynamic epsilon based on success rate
        dynamic_epsilon = self.epsilon
        if self.success_rate > 0.3:
            dynamic_epsilon *= 0.99  # Faster decay when performing well
        else:
            dynamic_epsilon *= 0.95  # Slower decay when struggling
        dynamic_epsilon = max(dynamic_epsilon, MIN_EPSILON)
        self.epsilon = dynamic_epsilon
        
        # Check if random action or best Q-value
        if random.random() < dynamic_epsilon:
            # Force movement if stuck at edges
            if state[0] < -0.8:  # Stuck at left
                return 1 if random.random() < 0.9 else random.randint(0, 2)
            elif state[0] > 0.8:  # Stuck at right
                return 0 if random.random() < 0.9 else random.randint(0, 2)
                
            # Aggressive enemy targeting
            if state[3] >= 0:  # If enemy position is valid
                distance = state[0] - state[3]
                normalized_distance = abs(distance) / 800  # Assuming SCREEN_WIDTH = 800
                
                # If far from enemy, prioritize movement
                if normalized_distance > 0.3:
                    return 1 if distance < -0.1 else 0
                
                # If close to enemy, consider shooting
                if normalized_distance < 0.1 and not is_bullet_active:
                    return 2  # SHOOT if well-aligned
            
            # If bullet is active, don't shoot
            if is_bullet_active:
                return random.randint(0, 1)  # LEFT or RIGHT
            return random.randint(0, 2)
        else:
            # Get Q-values from network
            with torch.no_grad():
                q_values = self.main_network(state_tensor)
            
            # Consider enemy position for exploitation
            if state[3] >= 0:  # If enemy position is valid
                distance = state[0] - state[3]
                if distance < -0.1 and q_values[0, 1] > q_values[0, 0] and q_values[0, 1] > q_values[0, 2]:
                    return 1  # RIGHT
                elif distance > 0.1 and q_values[0, 0] > q_values[0, 1] and q_values[0, 0] > q_values[0, 2]:
                    return 0  # LEFT
                elif abs(distance) < 0.1 and not is_bullet_active and q_values[0, 2] > q_values[0, 0] and q_values[0, 2] > q_values[0, 1]:
                    return 2  # SHOOT
            
            # If bullet is active, don't consider SHOOT action
            action_limit = 2 if is_bullet_active else 3
            actions = q_values[0, :action_limit]
            return torch.argmax(actions).item()
    
    def calculate_enhanced_reward(self, state, next_state, base_reward, is_hit=False):
        reward = base_reward
        
        # Hit reward and success tracking
        if is_hit:
            reward += 30.0  # HIT_REWARD from Arduino
            self.success_history[self.success_index] = 1
        else:
            reward -= 2.0  # MISS_PENALTY
            self.success_history[self.success_index] = 0
        
        self.success_index = (self.success_index + 1) % self.success_window
        self.success_rate = sum(self.success_history) / self.success_window
        
        # Edge penalty
        if state[0] < -0.8 or state[0] > 0.8:
            reward -= 6.0  # EDGE_PENALTY
        
        # Center reward
        if abs(state[0]) < 0.2:
            reward += 0.5  # CENTER_REWARD
        
        # Enemy targeting rewards
        if state[3] >= 0 and state[4] >= 0:
            distance = abs(state[0] - state[3])
            normalized_distance = distance / 800  # Assuming SCREEN_WIDTH = 800
            
            # Distance penalty
            if normalized_distance > 0.5:
                reward -= 1.0 * normalized_distance  # ENEMY_DISTANCE_PENALTY
            
            # Alignment reward
            if normalized_distance < 0.05:
                reward += 3.0 * 3.0  # ALIGNMENT_REWARD * 3.0
            elif normalized_distance < 0.2:
                reward += 3.0  # ALIGNMENT_REWARD
            
            # Direction reward
            position_change = state[0] - next_state[0]
            if (state[0] < state[3] and position_change > 0) or (state[0] > state[3] and position_change < 0):
                reward += 0.7 * (1.0 - normalized_distance)  # DIRECTION_REWARD
        
        # Stuck penalty
        if abs(state[0] - next_state[0]) < 0.01:
            reward -= 4.0  # STUCK_PENALTY
        
        # Scale reward based on success rate
        if self.success_rate > 0.3:
            reward *= 1.5  # Stronger boost when doing well
        elif self.success_rate < 0.1:
            reward *= 0.7  # Stronger reduction when doing poorly
        
        return reward
    
    def optimize_model(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        
        # Calculate adaptive learning rate
        current_lr = INITIAL_LEARNING_RATE * (LEARNING_RATE_DECAY ** self.steps)
        current_lr = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, current_lr))
        
        # Adjust learning rate based on success rate
        if self.success_rate > 0.3:
            current_lr *= 1.2  # Increase learning rate if doing well
        elif self.success_rate < 0.1:
            current_lr *= 0.8  # Decrease learning rate if doing poorly
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Sample batch
        transitions = self.memory.sample(BATCH_SIZE)
        
        # Convert batch to tensors
        batch_state = torch.FloatTensor([t[0] for t in transitions])
        batch_action = torch.LongTensor([[t[1]] for t in transitions])
        batch_reward = torch.FloatTensor([[t[2]] for t in transitions])
        batch_next_state = torch.FloatTensor([t[3] for t in transitions])
        
        # Compute Q values
        q_values = self.main_network(batch_state).gather(1, batch_action)
        
        # Compute next state values using Double DQN
        next_actions = self.main_network(batch_next_state).max(1)[1].unsqueeze(1)
        next_state_values = self.target_network(batch_next_state).gather(1, next_actions)
        
        # Compute target Q values
        target_q_values = batch_reward + DISCOUNT_FACTOR * next_state_values
        
        # Compute loss and update weights
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping as in Arduino
        torch.nn.utils.clip_grad_value_(self.main_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update steps and target network if needed
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
            if self.success_rate > 0.25:
                self.save_weights('live_weights.h')
            print(f"Steps: {self.steps}, Epsilon: {self.epsilon:.4f}, LR: {current_lr:.6f}, Success Rate: {self.success_rate:.3f}")
    
    def save_weights(self, filename):
        # Get weights as numpy arrays
        weights1 = self.main_network.fc1.weight.data.numpy()
        bias1 = self.main_network.fc1.bias.data.numpy()
        weights2 = self.main_network.fc2.weight.data.numpy()
        bias2 = self.main_network.fc2.bias.data.numpy()
        weights3 = self.main_network.fc3.weight.data.numpy()
        bias3 = self.main_network.fc3.bias.data.numpy()
        
        # Transpose weights to match Arduino format
        weights1 = weights1.T  # Arduino expects [input][hidden] format
        weights2 = weights2.T  # Arduino expects [hidden][hidden] format
        weights3 = weights3.T  # Arduino expects [hidden][output] format
        
        with open(filename, 'w') as f:
            f.write('#ifndef LIVE_WEIGHTS_H\n')
            f.write('#define LIVE_WEIGHTS_H\n\n')
            
            # Add constants
            f.write('const int INPUT_DIM = 5;\n')
            f.write('const int HIDDEN_DIM = 16;\n')
            f.write('const int OUTPUT_DIM = 3;\n\n')
            
            # First layer weights
            f.write('const float weights1[INPUT_DIM][HIDDEN_DIM] = {\n')
            for i in range(INPUT_DIM):
                f.write('    {')
                f.write(', '.join([f'{w:.6f}f' for w in weights1[i]]))
                f.write('},\n')
            f.write('};\n\n')
            
            # First layer bias
            f.write('const float bias1[HIDDEN_DIM] = {\n    ')
            f.write(', '.join([f'{b:.6f}f' for b in bias1]))
            f.write('\n};\n\n')
            
            # Second layer weights
            f.write('const float weights2[HIDDEN_DIM][HIDDEN_DIM] = {\n')
            for i in range(HIDDEN_DIM):
                f.write('    {')
                f.write(', '.join([f'{w:.6f}f' for w in weights2[i]]))
                f.write('},\n')
            f.write('};\n\n')
            
            # Second layer bias
            f.write('const float bias2[HIDDEN_DIM] = {\n    ')
            f.write(', '.join([f'{b:.6f}f' for b in bias2]))
            f.write('\n};\n\n')
            
            # Third layer weights
            f.write('const float weights3[HIDDEN_DIM][OUTPUT_DIM] = {\n')
            for i in range(HIDDEN_DIM):
                f.write('    {')
                f.write(', '.join([f'{w:.6f}f' for w in weights3[i]]))
                f.write('},\n')
            f.write('};\n\n')
            
            # Third layer bias
            f.write('const float bias3[OUTPUT_DIM] = {\n    ')
            f.write(', '.join([f'{b:.6f}f' for b in bias3]))
            f.write('\n};\n\n')
            
            f.write('#endif // LIVE_WEIGHTS_H\n')
        
        print(f"Weights saved to {filename}")

# Game controls help message
def print_game_controls():
    print("\n=== Game Controls ===")
    print("Note: The AI agent is controlling the game, but you can observe")
    print("Left/Right Arrow Keys: Manual player movement (if you want to test)")
    print("Space: Manual shooting (if you want to test)")
    print("Close Window: Exit training")
    print("====================\n")

# Training function that connects to the game environment
def train_dqn():
    # Create our wrapper for the game environment - start with rendering disabled
    env = GameEnvironment(render=False)
    agent = DQNAgent()
    
    num_episodes = 1000
    max_timesteps = 10000
    
    # Add display info
    print("Training started with rendering disabled for maximum speed.")
    print("Rendering will be enabled when success rate exceeds 0.25")
    print("Press Ctrl+C to stop training at any time.")
    
    for episode in range(num_episodes):
        state = env.reset()
        
        # Check if we should enable rendering based on success rate
        if agent.success_rate > 0.25 and not env.render:
            env.set_render(True)
        
        # Normalize state to match Arduino processing
        state[0] = (state[0] / 400.0) - 1.0  # Normalize player_x to [-1, 1]
        for i in range(1, 5):
            if state[i] >= 0:  # Only normalize if value is valid
                state[i] = state[i] / 800.0  # Normalize to [0, 1]
        
        episode_reward = 0
        is_bullet_active = False
        
        for t in range(max_timesteps):
            # Process Pygame events only if rendering
            if env.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            # Select action
            action = agent.select_action(state, is_bullet_active)
            
            try:
                # Take action and observe next state
                next_state, reward, done = env.step(action)
                
                # Add a small delay only if rendering
                if env.render:
                    time.sleep(0.01)  # 10ms delay
                
                # Update is_bullet_active flag
                is_bullet_active = (next_state[1] >= 0 and next_state[2] >= 0)
                
                # Normalize next_state
                normalized_next_state = next_state.copy()
                normalized_next_state[0] = (normalized_next_state[0] / 400.0) - 1.0
                for i in range(1, 5):
                    if normalized_next_state[i] >= 0:
                        normalized_next_state[i] = normalized_next_state[i] / 800.0
                
                # Calculate enhanced reward
                hit_detected = reward >= 10  # Hit gives reward >= 10 in the environment
                enhanced_reward = agent.calculate_enhanced_reward(state, normalized_next_state, reward, hit_detected)
                
                # Store transition in memory
                agent.memory.push(state, action, enhanced_reward, normalized_next_state)
                
                # Train the network
                agent.optimize_model()
                
                state = normalized_next_state
                episode_reward += reward
                
                if done:
                    break
                    
            except Exception as e:
                print(f"Error during training: {e}")
                # Try to continue with the next step
                continue
                
        # Also check after each episode if we should enable rendering
        if agent.success_rate > 0.25 and not env.render:
            env.set_render(True)
            
        # Decay epsilon
        agent.epsilon = max(FINAL_EPSILON, agent.epsilon * EPSILON_DECAY)
        
        # Show the screen for a few seconds at the end of each episode
        if not env.render:
            env.set_render(True)
        time.sleep(2)  # Show the screen for 2 seconds
        if env.render:
            env.set_render(False)

        # Save weights after every episode
        agent.save_weights('live_weights.h')
        
        # Print episode results
        print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Success: {agent.success_rate:.3f}")
        
    print("Training complete! Final weights saved to live_weights.h")

if __name__ == "__main__":
    try:
        train_dqn()
    except Exception as e:
        print(f"Training failed with error: {e}")