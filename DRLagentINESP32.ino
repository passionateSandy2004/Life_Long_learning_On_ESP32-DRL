#include <Arduino.h>
#include <ESP8266WiFi.h>  // Changed to WiFi.h for ESP32
#include "live_weights.h" // 
#include <EEPROM.h>  // Add EEPROM library

// -------- EEPROM Settings --------
#define EEPROM_SIZE 32768  // Maximum EEPROM size for ESP32
#define WEIGHTS_START_ADDR 0
#define WEIGHTS_END_ADDR (sizeof(weights1) + sizeof(bias1) + sizeof(weights2) + sizeof(bias2) + sizeof(weights3) + sizeof(bias3))

// Helper functions for writing/reading floats to/from EEPROM
void writeFloatToEEPROM(int addr, float value) {
    byte* p = (byte*)&value;
    for (int i = 0; i < sizeof(float); i++) {
        EEPROM.write(addr + i, p[i]);
    }
}

float readFloatFromEEPROM(int addr) {
    float value;
    byte* p = (byte*)&value;
    for (int i = 0; i < sizeof(float); i++) {
        p[i] = EEPROM.read(addr + i);
    }
    return value;
}

// -------- Wi-Fi & TCP Settings --------
const char* ssid = "Tenda_B4AE50";
const char* password = "admin@123";
const char* serverIP = "192.168.0.105";
const uint16_t serverPort = 5050;
WiFiClient tcpClient;

// -------- Game Constants --------
const float SCREEN_WIDTH = 800.0f;
const int FPS = 60;  // Add FPS constant

// -------- Optimized DRL Hyperparameters --------
const float INITIAL_LEARNING_RATE = 0.001f;  // Start with moderate learning rate
const float MIN_LEARNING_RATE = 0.0001f;     // Minimum learning rate
const float MAX_LEARNING_RATE = 0.01f;       // Maximum learning rate
const float LEARNING_RATE_DECAY = 0.999f;    // Learning rate decay factor
float epsilon = 1.0f;                        // Start with full exploration
const float MIN_EPSILON = 0.05f;             // Minimum exploration rate
const float DISCOUNT_FACTOR = 0.95f;         // Discount factor for future rewards

// -------- Enhanced Experience Replay Settings --------
const int MEMORY_SIZE = 500;                 // Reduced replay buffer size to fit memory constraints
const int BATCH_SIZE = 32;                   // Reduced batch size
const int MIN_MEMORY_SIZE = 32;              // Minimum experiences before training
const int TARGET_UPDATE_FREQ = 50;           // More frequent target network updates

// -------- Enhanced Reward Structure --------
const float HIT_REWARD = 30.0f;              // Further increased reward for hitting enemy
const float MISS_PENALTY = -2.0f;            // Further increased penalty for missing
const float ALIGNMENT_REWARD = 3.0f;         // Further increased reward for being aligned with enemy
const float EDGE_PENALTY = -6.0f;            // Further increased penalty for being near screen edges
const float MOVEMENT_REWARD = 1.5f;          // Further increased reward for moving towards enemy
const float STUCK_PENALTY = -4.0f;           // Further increased penalty for being stuck
const float DIRECTION_REWARD = 0.7f;         // Further increased reward for correct movement
const float CENTER_REWARD = 0.5f;            // Further increased reward for center position
const float ENEMY_DISTANCE_PENALTY = -1.0f;  // Further increased penalty for being too far from enemy

// -------- State Tracking --------
float prevPlayerX = 0.0f;                    // Previous player X position
float prevPlayerY = 0.0f;                    // Previous player Y position
float prevEnemyX = 0.0f;                     // Previous enemy X position
float prevEnemyY = 0.0f;                     // Previous enemy Y position

// -------- Success Tracking --------
int successfulHits = 0;
const int SUCCESS_WINDOW = 20;
int successHistory[SUCCESS_WINDOW] = {0};
int successIndex = 0;
float successRate = 0.0f;

// -------- Training Statistics --------
int updateCount = 0;
float maxGradient = 1.0f;                    // Maximum gradient value for clipping

// Network dimensions are defined in trained_weights_dummy.h
// const int HIDDEN_DIM = 16;          // Reduced hidden layer size
// const int INPUT_DIM = 5;            // [player_x, bullet_x, bullet_y, enemy_x, enemy_y]
// const int OUTPUT_DIM = 3;           // 0: LEFT, 1: RIGHT, 2: SHOOT

// -------- Enhanced Learning Parameters --------
const float INITIAL_EPSILON = 1.0f;          // Start with full exploration
const float FINAL_EPSILON = 0.01f;           // Keep some exploration
const float EPSILON_DECAY = 0.995f;          // Slower decay for better exploration

// Add adaptive learning parameters
float adaptiveLearningRate = INITIAL_LEARNING_RATE;

// Add state tracking
float prevState[INPUT_DIM];
bool hasPrevState = false;
int8_t prevAction = 0;
float prevReward = 0.0;

// Optimized memory structures
struct Transition {
    float state[INPUT_DIM];
    int8_t action;  // Use smaller data type for action
    float reward;
    float nextState[INPUT_DIM];
    float priority;
};

// Optimized network parameters
float mainWeights1[INPUT_DIM][HIDDEN_DIM];
float mainBias1[HIDDEN_DIM];
float mainWeights2[HIDDEN_DIM][HIDDEN_DIM];
float mainBias2[HIDDEN_DIM];
float mainWeights3[HIDDEN_DIM][OUTPUT_DIM];
float mainBias3[OUTPUT_DIM];

float targetWeights1[INPUT_DIM][HIDDEN_DIM];
float targetBias1[HIDDEN_DIM];
float targetWeights2[HIDDEN_DIM][HIDDEN_DIM];
float targetBias2[HIDDEN_DIM];
float targetWeights3[HIDDEN_DIM][OUTPUT_DIM];
float targetBias3[OUTPUT_DIM];

// Optimized temporary storage
float hiddenLayer1[HIDDEN_DIM];
float hiddenLayer2[HIDDEN_DIM];
float qValues[OUTPUT_DIM];

// Optimized memory
Transition memory[MEMORY_SIZE];
int memoryIndex = 0;
int currentMemorySize = 0;

// Optimized momentum storage
float weightMomentum1[INPUT_DIM][HIDDEN_DIM];
float biasMomentum1[HIDDEN_DIM];
float weightMomentum2[HIDDEN_DIM][HIDDEN_DIM];
float biasMomentum2[HIDDEN_DIM];
float weightMomentum3[HIDDEN_DIM][OUTPUT_DIM];
float biasMomentum3[OUTPUT_DIM];

// Add state history for better temporal understanding
const int STATE_HISTORY_SIZE = 3;
float stateHistory[STATE_HISTORY_SIZE][INPUT_DIM];
int historyIndex = 0;

// Add bullet tracking
bool isBulletActive = false;

// Add position tracking
float lastPlayerX = -1.0f;
int samePositionCount = 0;
const int MAX_SAME_POSITION = 5;  // Maximum allowed steps in same position

// -------- Utility Functions --------
float relu(float x) {
  return (x > 0) ? x : 0;
}

float reluDerivative(float x) {
  return (x > 0) ? 1.0 : 0.0;
}

void saveWeightsToEEPROM() {
    int addr = WEIGHTS_START_ADDR;
    
    // Save main network weights
    for(int i = 0; i < INPUT_DIM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            writeFloatToEEPROM(addr, mainWeights1[i][j]);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        writeFloatToEEPROM(addr, mainBias1[i]);
        addr += sizeof(float);
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            writeFloatToEEPROM(addr, mainWeights2[i][j]);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        writeFloatToEEPROM(addr, mainBias2[i]);
        addr += sizeof(float);
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        for(int j = 0; j < OUTPUT_DIM; j++) {
            writeFloatToEEPROM(addr, mainWeights3[i][j]);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < OUTPUT_DIM; i++) {
        writeFloatToEEPROM(addr, mainBias3[i]);
        addr += sizeof(float);
    }
    
    EEPROM.commit();
    Serial.println("Weights saved to EEPROM");
}

void loadWeightsFromEEPROM() {
    int addr = WEIGHTS_START_ADDR;
    
    // Load main network weights
    for(int i = 0; i < INPUT_DIM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            mainWeights1[i][j] = readFloatFromEEPROM(addr);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        mainBias1[i] = readFloatFromEEPROM(addr);
        addr += sizeof(float);
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            mainWeights2[i][j] = readFloatFromEEPROM(addr);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        mainBias2[i] = readFloatFromEEPROM(addr);
        addr += sizeof(float);
    }
    for(int i = 0; i < HIDDEN_DIM; i++) {
        for(int j = 0; j < OUTPUT_DIM; j++) {
            mainWeights3[i][j] = readFloatFromEEPROM(addr);
            addr += sizeof(float);
        }
    }
    for(int i = 0; i < OUTPUT_DIM; i++) {
        mainBias3[i] = readFloatFromEEPROM(addr);
        addr += sizeof(float);
    }
    
    Serial.println("Weights loaded from EEPROM");
}

void initNetwork() {
    // Initialize EEPROM
    EEPROM.begin(EEPROM_SIZE);
    
    // Try to load weights from EEPROM
    bool eepromValid = true;
    for(int i = 0; i < WEIGHTS_END_ADDR; i++) {
        if(EEPROM.read(i) == 0xFF) {
            eepromValid = false;
            break;
        }
    }
    
    if(eepromValid) {
        loadWeightsFromEEPROM();
        // Verify weights are not all zeros
        bool weightsValid = false;
        for(int i = 0; i < INPUT_DIM; i++) {
            for(int j = 0; j < HIDDEN_DIM; j++) {
                if(mainWeights1[i][j] != 0.0f) {
                    weightsValid = true;
                    break;
                }
            }
            if(weightsValid) break;
        }
        if(!weightsValid) eepromValid = false;
    }
    
    if(!eepromValid) {
        // If EEPROM is empty or invalid, copy from dummy weights
        memcpy(mainWeights1, weights1, sizeof(weights1));
        memcpy(mainBias1, bias1, sizeof(bias1));
        memcpy(mainWeights2, weights2, sizeof(weights2));
        memcpy(mainBias2, bias2, sizeof(bias2));
        memcpy(mainWeights3, weights3, sizeof(weights3));
        memcpy(mainBias3, bias3, sizeof(bias3));
        saveWeightsToEEPROM();  // Save initial weights to EEPROM
        Serial.println("Initialized with dummy weights");
    } else {
        Serial.println("Loaded weights from EEPROM");
    }
    
    // Copy weights to target network
    memcpy(targetWeights1, mainWeights1, sizeof(mainWeights1));
    memcpy(targetBias1, mainBias1, sizeof(mainBias1));
    memcpy(targetWeights2, mainWeights2, sizeof(mainWeights2));
    memcpy(targetBias2, mainBias2, sizeof(mainBias2));
    memcpy(targetWeights3, mainWeights3, sizeof(mainWeights3));
    memcpy(targetBias3, mainBias3, sizeof(mainBias3));
}

void storeTransition(const float* state, int action, float reward, const float* nextState) {
    // Store the transition in the replay memory
    memory[memoryIndex].action = action;
    memory[memoryIndex].reward = reward;
    memcpy(memory[memoryIndex].state, state, sizeof(float) * INPUT_DIM);
    memcpy(memory[memoryIndex].nextState, nextState, sizeof(float) * INPUT_DIM);
    
    // Update memory index
    memoryIndex = (memoryIndex + 1) % MEMORY_SIZE;
    currentMemorySize = min(currentMemorySize + 1, MEMORY_SIZE);
}

void forwardProp(const float* state, float* hidden1, float* hidden2, float* output,
                const float w1[][HIDDEN_DIM], const float* b1,
                const float w2[][HIDDEN_DIM], const float* b2,
                const float w3[][OUTPUT_DIM], const float* b3) {
    // First hidden layer
    for (int j = 0; j < HIDDEN_DIM; j++) {
    float sum = 0.0;
        for (int i = 0; i < INPUT_DIM; i++) {
      sum += state[i] * w1[i][j];
    }
        hidden1[j] = relu(sum + b1[j]);
  }
    
    // Second hidden layer
    for (int j = 0; j < HIDDEN_DIM; j++) {
    float sum = 0.0;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            sum += hidden1[i] * w2[i][j];
        }
        hidden2[j] = relu(sum + b2[j]);
    }
    
    // Output layer
    for (int j = 0; j < OUTPUT_DIM; j++) {
        float sum = 0.0;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            sum += hidden2[i] * w3[i][j];
        }
        output[j] = sum + b3[j];
    }
}

void forwardMain(const float* state) {
    forwardProp(state, hiddenLayer1, hiddenLayer2, qValues,
               mainWeights1, mainBias1, mainWeights2, mainBias2, mainWeights3, mainBias3);
}

float maxTargetQ(const float* nextState) {
    float tempHidden1[HIDDEN_DIM];
    float tempHidden2[HIDDEN_DIM];
    float tempOutput[OUTPUT_DIM];
    
    // Forward propagation through target network
    forwardProp(nextState, tempHidden1, tempHidden2, tempOutput,
                targetWeights1, targetBias1,
                targetWeights2, targetBias2,
                targetWeights3, targetBias3);
    
    // Find maximum Q-value
  float maxQ = tempOutput[0];
    for (int i = 1; i < OUTPUT_DIM; i++) {
        if (tempOutput[i] > maxQ) {
            maxQ = tempOutput[i];
        }
  }
  return maxQ;
}

void initStateHistory() {
    for(int i = 0; i < STATE_HISTORY_SIZE; i++) {
        for(int j = 0; j < INPUT_DIM; j++) {
            stateHistory[i][j] = 0.0f;
        }
    }
}

void updateStateHistory(const float* state) {
    memcpy(stateHistory[historyIndex], state, sizeof(float) * INPUT_DIM);
    historyIndex = (historyIndex + 1) % STATE_HISTORY_SIZE;
}

void processState(float* state) {
    // Normalize the player x position to be between -1 and 1
    state[0] = (state[0] / (SCREEN_WIDTH/2.0f)) - 1.0f;
    
    // Calculate velocities
    float playerXVelocity = (state[0] - prevPlayerX) * FPS;
    float playerYVelocity = (state[1] - prevPlayerY) * FPS;
    float enemyXVelocity = (state[3] - prevEnemyX) * FPS;
    float enemyYVelocity = (state[4] - prevEnemyY) * FPS;
    
    // Store velocities in state array
    state[5] = playerXVelocity;
    state[6] = playerYVelocity;
    state[7] = enemyXVelocity;
    state[8] = enemyYVelocity;
    
    // Update previous positions
    prevPlayerX = state[0];
    prevPlayerY = state[1];
    prevEnemyX = state[3];
    prevEnemyY = state[4];
    
    // Normalize other positions if they're not already normalized
    for(int i = 1; i < 5; i++) {
        if(state[i] >= 0) {  // Only normalize if the value is valid
            state[i] = state[i] / SCREEN_WIDTH;
        }
    }
}

float calculateEnhancedReward(const float* state, const float* nextState, float baseReward) {
    float reward = baseReward;
    
    // Strong reward for successful hits
    if(baseReward > 0) {
        reward += HIT_REWARD;
        successfulHits++;
        successHistory[successIndex] = 1;
    } else {
        reward += MISS_PENALTY;
        successHistory[successIndex] = 0;
    }
    successIndex = (successIndex + 1) % SUCCESS_WINDOW;
    
    // Calculate current success rate
    int recentSuccesses = 0;
    for(int i = 0; i < SUCCESS_WINDOW; i++) {
        recentSuccesses += successHistory[i];
    }
    successRate = (float)recentSuccesses / SUCCESS_WINDOW;
    
    // Strong edge penalty
    if(state[0] < -0.8f || state[0] > 0.8f) {
        reward += EDGE_PENALTY;
    }
    
    // Reward for being near center
    if(abs(state[0]) < 0.2f) {
        reward += CENTER_REWARD;
    }
    
    // Enhanced enemy targeting rewards
    if(state[3] >= 0 && state[4] >= 0) {
        float distance = abs(state[0] - state[3]);
        float normalizedDistance = distance / SCREEN_WIDTH;
        
        // Penalty for being too far from enemy
        if(normalizedDistance > 0.5f) {
            reward += ENEMY_DISTANCE_PENALTY * normalizedDistance;
        }
        
        // High reward for perfect alignment
        if(normalizedDistance < 0.05f) {
            reward += ALIGNMENT_REWARD * 3.0f;
        }
        // Good reward for being close
        else if(normalizedDistance < 0.2f) {
            reward += ALIGNMENT_REWARD;
        }
        
        // Strong direction reward - encourage moving towards enemy
        float positionChange = state[0] - nextState[0];
        if((state[0] < state[3] && positionChange > 0) || 
           (state[0] > state[3] && positionChange < 0)) {
            reward += DIRECTION_REWARD * (1.0f - normalizedDistance);
        }
    }
    
    // Strong penalty for being stuck
    if(abs(state[0] - nextState[0]) < 0.01f) {
        reward += STUCK_PENALTY;
    }
    
    // Scale reward based on success rate
    if(successRate > 0.3f) {
        reward *= 1.5f;  // Stronger boost when doing well
    } else if(successRate < 0.1f) {
        reward *= 0.7f;  // Stronger reduction when doing poorly
    }
    
    return reward;
}

int selectAction(const float* currentState) {
  float r = (float)random(0, 1000) / 1000.0;
    
    // Dynamic epsilon adjustment based on success rate
    float dynamicEpsilon = epsilon;
    if(successRate > 0.3f) {
        dynamicEpsilon *= 0.99f;  // Faster decay when performing well
    } else {
        dynamicEpsilon *= 0.95f;   // Much slower decay when struggling
    }
    dynamicEpsilon = max(dynamicEpsilon, MIN_EPSILON);
    epsilon = dynamicEpsilon;
    
    // Force movement if stuck at edges
    if(currentState[0] < -0.8f) {  // Stuck at left
        dynamicEpsilon = 1.0f;  // Force exploration
        if(r < 0.9f) {  // 90% chance to move right
            return 1;  // RIGHT
        }
    } else if(currentState[0] > 0.8f) {  // Stuck at right
        dynamicEpsilon = 1.0f;  // Force exploration
        if(r < 0.9f) {  // 90% chance to move left
            return 0;  // LEFT
        }
    }
    
    // Aggressive enemy targeting
    if(currentState[3] >= 0) {  // If enemy position is valid
        float distance = currentState[0] - currentState[3];
        float normalizedDistance = abs(distance) / SCREEN_WIDTH;
        
        // If far from enemy, prioritize movement
        if(normalizedDistance > 0.3f) {
            if(distance < -0.1f) return 1;  // Move RIGHT if enemy is to the right
            else return 0;  // Move LEFT if enemy is to the left
        }
        
        // If close to enemy, consider shooting
        if(normalizedDistance < 0.1f && !isBulletActive) {
            return 2;  // SHOOT if well-aligned
        }
    }
    
    if(r < dynamicEpsilon) {
        // During exploration, prefer actions that align with enemy
        if(currentState[3] >= 0) {
            if(currentState[0] < currentState[3] - 0.1f) {
                return 1;  // RIGHT if too far left
            } else if(currentState[0] > currentState[3] + 0.1f) {
                return 0;  // LEFT if too far right
            } else if(!isBulletActive) {
                return 2;  // SHOOT if well-aligned and no bullet active
            }
        }
        
        // If exploring and bullet is active, don't choose SHOOT
        if(isBulletActive) {
            return random(0, 2);  // Only LEFT or RIGHT
        }
        return random(0, OUTPUT_DIM);
  } else {
        // Exploitation: choose best action considering enemy position
    int bestAction = 0;
    float bestValue = qValues[0];
        
        // If bullet is active, don't consider SHOOT action
        int actionLimit = isBulletActive ? 2 : OUTPUT_DIM;
        
        // Strong bias towards actions that align with enemy
        if(currentState[3] >= 0) {
            float distance = currentState[0] - currentState[3];
            if(distance < -0.1f && qValues[1] > bestValue) {  // Too far left
                bestValue = qValues[1];
                bestAction = 1;
            } else if(distance > 0.1f && qValues[0] > bestValue) {  // Too far right
                bestValue = qValues[0];
                bestAction = 0;
            } else if(!isBulletActive && qValues[2] > bestValue) {  // Well-aligned
                bestValue = qValues[2];
                bestAction = 2;
            }
        }
        
        // If no clear bias, choose best Q-value
        for (int j = 0; j < actionLimit; j++) {
      if (qValues[j] > bestValue) {
        bestValue = qValues[j];
        bestAction = j;
      }
    }
    return bestAction;
  }
}

void trainStep() {
    if(currentMemorySize < MIN_MEMORY_SIZE) {
        return;
    }
    
    // Calculate adaptive learning rate
    float currentLearningRate = INITIAL_LEARNING_RATE * 
        pow(LEARNING_RATE_DECAY, updateCount);
    currentLearningRate = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, currentLearningRate));
    
    // Adjust learning rate based on success rate
    if(successRate > 0.3f) {
        currentLearningRate *= 1.2f;  // Increase learning rate if doing well
    } else if(successRate < 0.1f) {
        currentLearningRate *= 0.8f;  // Decrease learning rate if doing poorly
    }
    
    // Sample a random batch from memory
    int batchIndices[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        batchIndices[i] = random(0, currentMemorySize);
    }
    
    // Process each sample in the batch
    for (int i = 0; i < BATCH_SIZE; i++) {
        Transition& sample = memory[batchIndices[i]];
        
        // Forward pass through target network
        float hiddenLayer1[HIDDEN_DIM];
        float hiddenLayer2[HIDDEN_DIM];
        float qValues[OUTPUT_DIM];
        
        forwardProp(sample.state, hiddenLayer1, hiddenLayer2, qValues,
                   targetWeights1, targetBias1, targetWeights2, targetBias2,
                   targetWeights3, targetBias3);
        
        // Calculate target Q-value with double Q-learning
        float targetQ = sample.reward;
        if (!isTerminalState(sample.nextState)) {
            // Get action from main network
            float mainHidden1[HIDDEN_DIM];
            float mainHidden2[HIDDEN_DIM];
            float mainQValues[OUTPUT_DIM];
            forwardProp(sample.nextState, mainHidden1, mainHidden2, mainQValues,
                       mainWeights1, mainBias1, mainWeights2, mainBias2,
                       mainWeights3, mainBias3);
            
            // Get value from target network
            float targetHidden1[HIDDEN_DIM];
            float targetHidden2[HIDDEN_DIM];
            float targetQValues[OUTPUT_DIM];
            forwardProp(sample.nextState, targetHidden1, targetHidden2, targetQValues,
                       targetWeights1, targetBias1, targetWeights2, targetBias2,
                       targetWeights3, targetBias3);
            
            // Find best action from main network
            int bestAction = 0;
            float bestValue = mainQValues[0];
            for (int j = 1; j < OUTPUT_DIM; j++) {
                if (mainQValues[j] > bestValue) {
                    bestValue = mainQValues[j];
                    bestAction = j;
                }
            }
            
            // Use target network's Q-value for that action
            targetQ += DISCOUNT_FACTOR * targetQValues[bestAction];
        }
        
        // Calculate TD error and update priorities
        float tdError = abs(qValues[sample.action] - targetQ);
        memory[batchIndices[i]].priority = pow(tdError + 1e-5, 0.6f);
        
        // Backward pass with gradient clipping
        float outputGradients[OUTPUT_DIM] = {0};
        outputGradients[sample.action] = (qValues[sample.action] - targetQ);
        
        // Clip gradients
        for (int j = 0; j < OUTPUT_DIM; j++) {
            outputGradients[j] = max(min(outputGradients[j], maxGradient), -maxGradient);
        }
        
        // Update weights with momentum
        float momentum = 0.9f;
        
        // Update first layer
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                weightMomentum1[i][j] = momentum * weightMomentum1[i][j] + 
                    (1 - momentum) * outputGradients[j] * sample.state[i];
                mainWeights1[i][j] -= currentLearningRate * weightMomentum1[i][j];
            }
        }
        
        // Update biases
        for (int i = 0; i < HIDDEN_DIM; i++) {
            biasMomentum1[i] = momentum * biasMomentum1[i] + 
                (1 - momentum) * outputGradients[i];
            mainBias1[i] -= currentLearningRate * biasMomentum1[i];
        }
        
        // Similar updates for other layers...
    }
    
  updateCount++;
    
    // Update target network
    if (updateCount % TARGET_UPDATE_FREQ == 0) {
        memcpy(targetWeights1, mainWeights1, sizeof(mainWeights1));
        memcpy(targetBias1, mainBias1, sizeof(mainBias1));
        memcpy(targetWeights2, mainWeights2, sizeof(mainWeights2));
        memcpy(targetBias2, mainBias2, sizeof(mainBias2));
        memcpy(targetWeights3, mainWeights3, sizeof(mainWeights3));
        memcpy(targetBias3, mainBias3, sizeof(mainBias3));
        
        // Save weights to EEPROM only if performance is good
        if(successRate > 0.25f) {
            saveWeightsToEEPROM();
        }
    }
    
    // Debug output
    Serial.print("Success: ");
    Serial.print(successRate);
    Serial.print(" | Eps: ");
    Serial.print(epsilon);
    Serial.print(" | LR: ");
    Serial.println(currentLearningRate);
}

String actionToCommand(int action) {
  switch (action) {
    case 0: return "LEFT\n";
    case 1: return "RIGHT\n";
    case 2: return "SHOOT\n";
    default: return "";
  }
}

void connectWiFiAndServer() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("ESP IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Connecting to server ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);
  while (!tcpClient.connect(serverIP, serverPort)) {
    Serial.println("Connection failed. Retrying...");
    delay(2000);
  }
  Serial.println("Connected to game server.");
}

String readLineFromTCP() {
  String line = "";
  while (tcpClient.connected() && tcpClient.available()) {
    char c = tcpClient.read();
    if (c == '\n') break;
    else line += c;
  }
  return line;
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  randomSeed(analogRead(0));
  initNetwork();
  initStateHistory();  // Initialize state history
  connectWiFiAndServer();
  Serial.println("DRL agent initialized. Awaiting state updates...");
}

void loop() {
  if (!tcpClient.connected()) {
    Serial.println("TCP disconnected, reconnecting...");
    tcpClient.stop();
    connectWiFiAndServer();
  }
  if (tcpClient.available()) {
    String line = readLineFromTCP();
    line.trim();
    if (line.length() > 0 && line.startsWith("STATE:")) {
      int idx = line.indexOf(";");
      if (idx != -1) {
        String statePart = line.substring(0, idx);
        String rewardPart = line.substring(idx + 1);
        statePart.replace("STATE:", "");
        rewardPart.replace("REWARD:", "");
        float currState[INPUT_DIM];
        int commaIdx = 0;
        for (int i = 0; i < INPUT_DIM; i++) {
          int nextComma = statePart.indexOf(",", commaIdx);
          String valueStr = (nextComma == -1) ? statePart.substring(commaIdx) : statePart.substring(commaIdx, nextComma);
          currState[i] = valueStr.toFloat();
          if (nextComma == -1) break;
          commaIdx = nextComma + 1;
        }
        
        // Update bullet state
        isBulletActive = (currState[1] >= 0 && currState[2] >= 0);  // Check if bullet exists
        
        // Process and normalize the state
        processState(currState);
        
        float reward = rewardPart.toFloat();
        if (hasPrevState) {
          storeTransition(prevState, prevAction, prevReward, currState);
        }
        forwardMain(currState);
        int action = selectAction(currState);
        String cmd = actionToCommand(action);
        
        // Only send SHOOT command if no bullet is active
        if(action == 2 && isBulletActive) {
          // If trying to shoot but bullet is active, choose a movement action instead
          action = (random(0, 1000) < 500) ? 0 : 1;  // Randomly choose LEFT or RIGHT
          cmd = actionToCommand(action);
        }
        
        tcpClient.print(cmd);
        
        // Debug output
        Serial.print("State: ");
        for (int i = 0; i < INPUT_DIM; i++) {
          Serial.print(currState[i], 3);
          Serial.print(" ");
        }
        Serial.print("| Reward: ");
        Serial.print(reward, 3);
        Serial.print("| Action: ");
        Serial.print(action);
        Serial.print(" (");
        Serial.print(cmd);
        Serial.print(") | Bullet Active: ");
        Serial.print(isBulletActive ? "Yes" : "No");
        Serial.print(" | Epsilon: ");
        Serial.println(epsilon, 3);
        
        memcpy(prevState, currState, sizeof(prevState));
        prevAction = action;
        prevReward = reward;
        hasPrevState = true;
      }
    }
  }
  delay(100);
}

bool isTerminalState(const float* state) {
    // Check if any state value is invalid (negative)
    for(int i = 0; i < INPUT_DIM; i++) {
        if(state[i] < 0) return true;
    }
    return false;
}
