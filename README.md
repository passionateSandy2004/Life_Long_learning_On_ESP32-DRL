# DRL Agent for ESP32 Life Long learning - Shooting Game

This project implements a Deep Reinforcement Learning (DRL) agent that learns to play a shooting game on an ESP32 microcontroller. The agent uses a Deep Q-Network (DQN) to learn optimal strategies for moving and shooting in a 2D game environment.

## Project Structure

The project consists of three main components:

1. **Python Training Environment** (`Train_IN_Py.py`):
   - Implements the game environment using Pygame
   - Contains the DQN agent implementation
   - Handles training and weight generation
   - Saves trained weights in a format compatible with ESP32

2. **Game Environment** (`RL_env.py`):
   - Implements the actual game mechanics
   - Handles rendering and game state
   - Provides TCP server for communication with ESP32
   - Manages game objects (player, bullets, enemies)

3. **ESP32 Implementation** (`DRLagentINESP32.ino`):
   - Runs the trained DQN on the ESP32
   - Communicates with the game environment via TCP
   - Implements the neural network inference
   - Handles weight updates and learning

## Key Features

- **Adaptive Learning**: The agent uses dynamic learning rates and exploration rates based on performance
- **Experience Replay**: Implements prioritized experience replay for better learning
- **Double DQN**: Uses target network for more stable learning
- **Enhanced Rewards**: Sophisticated reward structure for better learning
- **Real-time Training**: Can train and update weights while playing
- **Cross-Platform**: Works between Python and ESP32

## Neural Network Architecture

The DQN consists of:
- Input Layer: 5 neurons (player_x, bullet_x, bullet_y, enemy_x, enemy_y)
- Hidden Layer 1: 16 neurons with ReLU activation
- Hidden Layer 2: 16 neurons with ReLU activation
- Output Layer: 3 neurons (LEFT, RIGHT, SHOOT)

## Training Parameters

- Learning Rate: 0.001 (adaptive)
- Discount Factor: 0.95
- Epsilon (exploration): 1.0 to 0.01
- Memory Size: 500
- Batch Size: 32
- Target Update Frequency: 50

## Reward Structure

The agent receives rewards for:
- Hitting enemies (+30.0)
- Being aligned with enemies (+3.0)
- Moving towards enemies (+0.7)
- Staying near center (+0.5)

And penalties for:
- Missing shots (-2.0)
- Being near edges (-6.0)
- Being stuck (-4.0)
- Being too far from enemies (-1.0)

## Setup and Usage

1. **Python Environment Setup**:
   ```bash
   pip install pygame numpy torch
   ```

2. **ESP32 Setup**:
   - Install Arduino IDE
   - Install ESP32 board support
   - Install required libraries (WiFi.h, EEPROM.h)

3. **Training**:
   ```bash
   python Train_IN_Py.py
   ```
   - The training will start in headless mode
   - Rendering will be enabled when success rate exceeds 0.25
   - Weights will be saved to `live_weights.h` periodically

4. **Running on ESP32**:
   - Upload the ESP32 code
   - Connect to the same WiFi network as the training environment
   - The ESP32 will automatically connect to the game environment

## Network Communication

- TCP Port: 5050
- State Format: "STATE:player_x,bullet_x,bullet_y,enemy_x,enemy_y;REWARD:value"
- Action Format: "LEFT\n", "RIGHT\n", or "SHOOT\n"

## Performance Monitoring

The system tracks:
- Success rate (hits in last 20 actions)
- Learning rate (adaptive)
- Epsilon (exploration rate)
- Training steps
- Episode rewards

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request
 
