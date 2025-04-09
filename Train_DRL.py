# train_drl_agent.py
import pygame
import random, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------- Game Environment Settings ---------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

PLAYER_WIDTH = 50
PLAYER_HEIGHT = 30
PLAYER_SPEED = 5

BULLET_WIDTH = 5
BULLET_HEIGHT = 10
BULLET_SPEED = 7

ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_SPEED = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)

# --------------------- DRL Hyperparameters ---------------------
LEARNING_RATE = 0.01
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99
TARGET_UPDATE = 25
MEMORY_SIZE = 200
BATCH_SIZE = 32

# Enhanced learning parameters
EXPLORATION_BOOST = 0.5
STUCK_PENALTY = -0.5
ALIGNMENT_REWARD = 0.2
BULLET_PROXIMITY_REWARD = 0.3
MOVEMENT_REWARD = 0.1
REWARD_SCALING = 2.0

# Adaptive learning parameters
MIN_LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.999
SUCCESS_THRESHOLD = 5

# --------------------- Network Architecture ---------------------
INPUT_DIM = 5     # [player_x, bullet_x, bullet_y, enemy_x, enemy_y]
HIDDEN_DIM = 32   # Must match ESP32's HIDDEN_DIM
OUTPUT_DIM = 3    # [LEFT, RIGHT, SHOOT]

# --------------------- DRL Agent Network ---------------------
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        # Ensure the dimensions match exactly with ESP32
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights to match ESP32's expected format
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --------------------- DRL Agent Class ---------------------
class DQNAgent:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.adaptive_learning_rate = LEARNING_RATE
        self.successful_hits = 0
        self.stuck_count = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon + EXPLORATION_BOOST
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.output_dim)]], 
                              device=self.device, dtype=torch.long)
    
    def calculate_enhanced_reward(self, state, next_state, base_reward):
        reward = base_reward * REWARD_SCALING
        
        # Movement reward with velocity consideration
        position_change = abs(state[0] - next_state[0])
        velocity = position_change / (1.0/FPS)
        reward += MOVEMENT_REWARD * velocity
        
        # Alignment reward with distance consideration
        if state[3] >= 0 and state[4] >= 0:
            distance = abs(state[0] - state[3])
            normalized_distance = distance / SCREEN_WIDTH
            alignment = 1.0 - normalized_distance
            reward += ALIGNMENT_REWARD * alignment
            
            if normalized_distance < 0.1:
                reward += ALIGNMENT_REWARD * 2.0
        
        # Bullet proximity reward
        if state[1] >= 0 and state[2] >= 0:
            bullet_dist = math.sqrt((state[1] - state[3])**2 + (state[2] - state[4])**2)
            max_dist = math.sqrt(2.0)
            proximity = 1.0 - (bullet_dist / max_dist)
            reward += BULLET_PROXIMITY_REWARD * proximity
            
            if bullet_dist < 0.2:
                reward += BULLET_PROXIMITY_REWARD * 2.0
        
        # Stuck penalty with time consideration
        if position_change < 0.01:
            reward += STUCK_PENALTY
            self.stuck_count += 1
            if self.stuck_count > 10:
                reward += STUCK_PENALTY * 2.0
                self.stuck_count = 0
        else:
            self.stuck_count = 0
        
        return reward
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample with priority
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.param_groups[0]['lr'] = self.adaptive_learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update adaptive learning rate
        if self.successful_hits >= SUCCESS_THRESHOLD:
            self.adaptive_learning_rate *= LEARNING_RATE_DECAY
            self.adaptive_learning_rate = max(self.adaptive_learning_rate, MIN_LEARNING_RATE)
            self.successful_hits = 0
        else:
            self.adaptive_learning_rate = min(self.adaptive_learning_rate * 1.01, MAX_LEARNING_RATE)
        
        return loss.item()
    
    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action.item())
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Calculate enhanced reward
                enhanced_reward = self.calculate_enhanced_reward(
                    state.squeeze().cpu().numpy(),
                    next_state.squeeze().cpu().numpy(),
                    reward
                )
                reward = torch.tensor([enhanced_reward], device=self.device)
                
                # Store transition in memory
                self.memory.append((state, action, reward, next_state))
                
                # Move to next state
                state = next_state
                total_reward += reward.item()
                
                # Perform one step of optimization
                loss = self.optimize_model()
                
                # Update target network
                if self.steps_done % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                self.steps_done += 1
                
                # Update epsilon
                self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, "
                      f"Learning Rate: {self.adaptive_learning_rate:.6f}")
    
    def save_weights(self, filename):
        try:
            # Convert weights to numpy arrays and ensure correct dimensions
            # Note: PyTorch stores weights in (output_dim, input_dim) format
            # We need to transpose them to match ESP32's (input_dim, output_dim) format
            weights = {
                'fc1.weight': self.policy_net.fc1.weight.data.cpu().numpy().T.tolist(),  # Transpose to match ESP32
                'fc1.bias': self.policy_net.fc1.bias.data.cpu().numpy().tolist(),
                'fc2.weight': self.policy_net.fc2.weight.data.cpu().numpy().T.tolist(),  # Transpose to match ESP32
                'fc2.bias': self.policy_net.fc2.bias.data.cpu().numpy().tolist(),
                'fc3.weight': self.policy_net.fc3.weight.data.cpu().numpy().T.tolist(),  # Transpose to match ESP32
                'fc3.bias': self.policy_net.fc3.bias.data.cpu().numpy().tolist()
            }
            
            # Verify dimensions
            assert len(weights['fc1.weight']) == INPUT_DIM, f"fc1.weight dimension mismatch: expected {INPUT_DIM}, got {len(weights['fc1.weight'])}"
            assert len(weights['fc1.weight'][0]) == HIDDEN_DIM, f"fc1.weight dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc1.weight'][0])}"
            assert len(weights['fc1.bias']) == HIDDEN_DIM, f"fc1.bias dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc1.bias'])}"
            
            assert len(weights['fc2.weight']) == HIDDEN_DIM, f"fc2.weight dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc2.weight'])}"
            assert len(weights['fc2.weight'][0]) == HIDDEN_DIM, f"fc2.weight dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc2.weight'][0])}"
            assert len(weights['fc2.bias']) == HIDDEN_DIM, f"fc2.bias dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc2.bias'])}"
            
            assert len(weights['fc3.weight']) == HIDDEN_DIM, f"fc3.weight dimension mismatch: expected {HIDDEN_DIM}, got {len(weights['fc3.weight'])}"
            assert len(weights['fc3.weight'][0]) == OUTPUT_DIM, f"fc3.weight dimension mismatch: expected {OUTPUT_DIM}, got {len(weights['fc3.weight'][0])}"
            assert len(weights['fc3.bias']) == OUTPUT_DIM, f"fc3.bias dimension mismatch: expected {OUTPUT_DIM}, got {len(weights['fc3.bias'])}"
            
            # Save weights to JSON file
            with open(filename, 'w') as f:
                json.dump(weights, f)
            
            # Generate C header file
            self.generate_header_file(weights)
            print(f"Weights saved successfully to {filename} and trained_weights.h")
        except Exception as e:
            print(f"Error saving weights: {str(e)}")

    def generate_header_file(self, weights):
        try:
            with open('trained_weights.h', 'w') as f:
                f.write('#ifndef TRAINED_WEIGHTS_H\n')
                f.write('#define TRAINED_WEIGHTS_H\n\n')
                
                f.write(f'#define INPUT_DIM {INPUT_DIM}\n')
                f.write(f'#define HIDDEN_DIM {HIDDEN_DIM}\n')
                f.write(f'#define OUTPUT_DIM {OUTPUT_DIM}\n\n')
                
                # First layer weights and bias
                f.write('const float weights1[INPUT_DIM][HIDDEN_DIM] = {\n')
                for i in range(INPUT_DIM):
                    f.write('    {')
                    f.write(', '.join([f'{w:.6f}f' for w in weights['fc1.weight'][i]]))
                    f.write('},\n')
                f.write('};\n\n')
                
                f.write('const float bias1[HIDDEN_DIM] = {\n    ')
                f.write(', '.join([f'{b:.6f}f' for b in weights['fc1.bias']]))
                f.write('\n};\n\n')
                
                # Second layer weights and bias
                f.write('const float weights2[HIDDEN_DIM][HIDDEN_DIM] = {\n')
                for i in range(HIDDEN_DIM):
                    f.write('    {')
                    f.write(', '.join([f'{w:.6f}f' for w in weights['fc2.weight'][i]]))
                    f.write('},\n')
                f.write('};\n\n')
                
                f.write('const float bias2[HIDDEN_DIM] = {\n    ')
                f.write(', '.join([f'{b:.6f}f' for b in weights['fc2.bias']]))
                f.write('\n};\n\n')
                
                # Third layer weights and bias
                f.write('const float weights3[HIDDEN_DIM][OUTPUT_DIM] = {\n')
                for i in range(HIDDEN_DIM):
                    f.write('    {')
                    f.write(', '.join([f'{w:.6f}f' for w in weights['fc3.weight'][i]]))
                    f.write('},\n')
                f.write('};\n\n')
                
                f.write('const float bias3[OUTPUT_DIM] = {\n    ')
                f.write(', '.join([f'{b:.6f}f' for b in weights['fc3.bias']]))
                f.write('\n};\n\n')
                
                f.write('#endif\n')
            print("Header file generated successfully")
        except Exception as e:
            print(f"Error generating header file: {str(e)}")

# --------------------- Pygame Sprites ---------------------
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.midbottom = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10)
        self.speed = 0
    def update(self):
        self.rect.x += self.speed
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BULLET_WIDTH, BULLET_HEIGHT))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed = -BULLET_SPEED  # Negative because bullets move upward

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:  # Remove bullet if it goes off screen
            self.kill()

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((ENEMY_WIDTH, ENEMY_HEIGHT))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, SCREEN_WIDTH - ENEMY_WIDTH)
        self.rect.y = 0
    def update(self):
        self.rect.y += ENEMY_SPEED
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

# --------------------- Game Environment Class ---------------------
class GameEnv:
    def __init__(self, render=False):
        self.render = render
        pygame.init()
        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shooting Game Environment")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.reset()
    
    def reset(self):
        self.all_sprites = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.player = Player()
        self.all_sprites.add(self.player)
        enemy = Enemy()
        self.all_sprites.add(enemy)
        self.enemies.add(enemy)
        return self.get_state()
    
    def get_state(self):
        # State: player_x, bullet_x, bullet_y, enemy_x, enemy_y
        player_x = float(self.player.rect.x)
        
        # Get the closest bullet to the enemy
        closest_bullet_x = -1.0
        closest_bullet_y = -1.0
        min_dist = float('inf')
        
        for bullet in self.bullets:
            if self.enemies:
                enemy = list(self.enemies)[0]
                dist = math.hypot(bullet.rect.centerx - enemy.rect.centerx,
                                bullet.rect.centery - enemy.rect.centery)
                if dist < min_dist:
                    min_dist = dist
                    closest_bullet_x = float(bullet.rect.x)
                    closest_bullet_y = float(bullet.rect.y)
        
        if self.enemies:
            enemy = list(self.enemies)[0]
            enemy_x = float(enemy.rect.x)
            enemy_y = float(enemy.rect.y)
        else:
            enemy_x, enemy_y = -1.0, -1.0
            
        return np.array([player_x, closest_bullet_x, closest_bullet_y, enemy_x, enemy_y], dtype=np.float32)
    
    def step(self, action):
        # Actions: 0 = LEFT, 1 = RIGHT, 2 = SHOOT
        if action == 0:
            self.player.speed = -PLAYER_SPEED
        elif action == 1:
            self.player.speed = PLAYER_SPEED
        elif action == 2:
            # Always allow shooting, don't check for existing bullets
            bullet = Bullet(self.player.rect.centerx, self.player.rect.top)
            self.all_sprites.add(bullet)
            self.bullets.add(bullet)
            
        # Update game elements
        self.all_sprites.update()
        reward = 0
        
        # Check bullet-enemy collisions
        hits = pygame.sprite.groupcollide(self.bullets, self.enemies, True, True)
        if hits:
            reward += 10
            enemy = Enemy()
            self.all_sprites.add(enemy)
            self.enemies.add(enemy)
            
        # Penalize enemy if it goes off screen
        for enemy in self.enemies:
            if enemy.rect.bottom >= SCREEN_HEIGHT:
                reward -= 5
                enemy.kill()
                enemy = Enemy()
                self.all_sprites.add(enemy)
                self.enemies.add(enemy)
                
        # Extra reward for alignment and bullet proximity
        if self.enemies:
            enemy = list(self.enemies)[0]
            alignment_reward = 1.0 - (abs(self.player.rect.centerx - enemy.rect.centerx) / SCREEN_WIDTH)
            if self.bullets:
                # Calculate reward based on all bullets
                total_bullet_reward = 0
                for bullet in self.bullets:
                    dist = math.hypot(bullet.rect.centerx - enemy.rect.centerx,
                                    bullet.rect.centery - enemy.rect.centery)
                    max_dist = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
                    bullet_reward = 1.0 - (dist / max_dist)
                    total_bullet_reward += bullet_reward
                bullet_reward = total_bullet_reward / len(self.bullets)  # Average bullet reward
            else:
                bullet_reward = 0
            reward += 2 * alignment_reward + 2 * bullet_reward
            
        next_state = self.get_state()
        done = False  # endless training episodes
        
        if self.render:
            self.render_screen(reward)
        self.clock.tick(FPS)
        return next_state, reward, done
    
    def render_screen(self, reward):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        info = [
            f"State: {self.get_state()}",
            f"Reward: {reward:.2f}",
            f"Bullets: {len(self.bullets)}"
        ]
        for idx, line in enumerate(info):
            text = self.font.render(line, True, WHITE)
            self.screen.blit(text, (10, 10 + idx * 20))
        pygame.display.flip()

# --------------------- Main Training Loop ---------------------
def main():
    try:
        # Initialize environment and agent
        print("\n=== Initializing Training Environment ===")
        print("Initializing environment...")
        env = GameEnv(render=False)
        print("Environment initialized successfully")
        
        print("\n=== Initializing DQN Agent ===")
        print("Initializing DQN agent...")
        agent = DQNAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        print("Agent initialized successfully")
        
        # Print initial parameters
        print("\n=== Training Parameters ===")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Memory Size: {MEMORY_SIZE}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Target Update Frequency: {TARGET_UPDATE}")
        print(f"Epsilon Start: {EPSILON_START}")
        print(f"Epsilon End: {EPSILON_END}")
        print(f"Epsilon Decay: {EPSILON_DECAY}")
        print(f"Gamma: {GAMMA}")
        print(f"Input Dimension: {INPUT_DIM}")
        print(f"Hidden Dimension: {HIDDEN_DIM}")
        print(f"Output Dimension: {OUTPUT_DIM}")
        
        # Train the agent
        print("\n=== Starting Training ===")
        total_episodes = 5  # Reduced to 10 episodes
        max_steps = 1000  # Reduced max steps
        best_reward = float('-inf')
        start_time = time.time()
        total_steps = 0
        total_rewards = []
        successful_hits = 0
        last_save_step = 0
        save_interval = 1000  # Save weights every 1000 steps
        
        for episode in range(total_episodes):
            try:
                print(f"\n=== Episode {episode + 1}/{total_episodes} ===")
                state = env.reset()
                state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                episode_reward = 0
                done = False
                step = 0
                episode_hits = 0
                
                while not done and total_steps < max_steps:
                    try:
                        # Select and perform action
                        action = agent.select_action(state)
                        next_state, reward, done = env.step(action.item())
                        
                        # Convert to tensor
                        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                        reward = torch.tensor([reward], device=agent.device)
                        
                        # Store transition
                        agent.memory.append((state, action, reward, next_state))
                        
                        # Move to next state
                        state = next_state
                        episode_reward += reward.item()
                        
                        # Track hits
                        if reward.item() > 0:
                            episode_hits += 1
                            successful_hits += 1
                        
                        # Optimize model
                        if len(agent.memory) >= BATCH_SIZE:
                            loss = agent.optimize_model()
                        
                        # Update target network
                        if step % TARGET_UPDATE == 0:
                            agent.target_net.load_state_dict(agent.policy_net.state_dict())
                        
                        step += 1
                        total_steps += 1
                        
                        # Update epsilon
                        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
                        
                        # Save weights periodically
                        if total_steps - last_save_step >= save_interval:
                            print(f"\n=== Saving Weights at Step {total_steps} ===")
                            agent.save_weights(f'trained_weights_step_{total_steps}.json')
                            last_save_step = total_steps
                    
                    except Exception as e:
                        print(f"Error in step {step}: {str(e)}")
                        continue
                
                # Track best performance and save best weights
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"\n=== New Best Performance ===")
                    print(f"Episode {episode}: New best reward = {best_reward:.2f}")
                    print(f"Epsilon: {agent.epsilon:.3f}")
                    print(f"Memory Size: {len(agent.memory)}")
                    print(f"Successful Hits: {episode_hits}")
                    print("Saving best weights...")
                    agent.save_weights('best_weights.json')
                
                # Store episode reward
                total_rewards.append(episode_reward)
                
                # Print progress after each episode
                elapsed_time = time.time() - start_time
                episodes_per_second = (episode + 1) / elapsed_time
                avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards)) if total_rewards else 0
                print(f"\n=== Training Progress ===")
                print(f"Episode: {episode}/{total_episodes}")
                print(f"Total Steps: {total_steps}")
                print(f"Episode Reward: {episode_reward:.2f}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Memory Size: {len(agent.memory)}")
                print(f"Successful Hits: {episode_hits}")
                print(f"Total Successful Hits: {successful_hits}")
                print(f"Training Speed: {episodes_per_second:.2f} eps/s")
                print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
                
                # Check if we've reached max steps
                if total_steps >= max_steps:
                    print("\n=== Maximum Steps Reached ===")
                    break
                
            except Exception as e:
                print(f"\nError in episode {episode}: {str(e)}")
                continue
        
        # Print final statistics
        print("\n=== Training Complete ===")
        print(f"Total Episodes: {episode + 1}")
        print(f"Total Steps: {total_steps}")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Average Reward: {sum(total_rewards)/len(total_rewards) if total_rewards else 0:.2f}")
        print(f"Total Successful Hits: {successful_hits}")
        print(f"Final Epsilon: {agent.epsilon:.3f}")
        
        # Save final weights
        print("\nSaving final weights...")
        agent.save_weights('final_weights.json')
        print("Weights saved to final_weights.json and trained_weights.h")
        
    except Exception as e:
        print(f"\nError in main training loop: {str(e)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
