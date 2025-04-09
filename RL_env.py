"""
Improved Shooting Game Environment with Dynamic Rendering Mode

This version runs in headless (training) mode initially (without rendering) 
and then, after a specified number of updates, it switches to rendering mode.
It still sends state updates over the network for training the DRL agent.
"""

import pygame
import socket
import threading
import queue
import time
import random
import math
import numpy as np

# --------------------- Networking Settings ---------------------
TCP_PORT = 5050  # Must match DRL agent code
BUFFER_SIZE = 1024

# Thread-safe queue for commands from the DRL agent
command_queue = queue.Queue()

# Global connection variables with a lock for thread safety
client_conn = None
conn_lock = threading.Lock()
connection_status = "Waiting for connection..."

def get_ip():
    """Return the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

SERVER_IP = get_ip()

# --------------------- Network Thread ---------------------
def network_thread():
    """
    Sets up a TCP server that listens for incoming connections from the DRL agent.
    Received newline-terminated commands are placed into the command_queue.
    """
    global client_conn, connection_status
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", TCP_PORT))
    s.listen(1)
    print(f"TCP server listening on {SERVER_IP}:{TCP_PORT}")

    while True:
        try:
            conn, addr = s.accept()
            with conn_lock:
                client_conn = conn
            connection_status = f"Connected: {addr[0]}"
            print(f"Connected by {addr}")
            data_buffer = ""
            while True:
                data = conn.recv(BUFFER_SIZE)
                if not data:
                    print("Connection closed by client.")
                    connection_status = "Waiting for connection..."
                    with conn_lock:
                        client_conn = None
                    conn.close()
                    break
                data_buffer += data.decode("utf-8")
                while "\n" in data_buffer:
                    line, data_buffer = data_buffer.split("\n", 1)
                    line = line.strip().upper()
                    if line:
                        print("Received command from agent:", line)
                        command_queue.put(line)
        except Exception as e:
            print("Network error:", e)
            connection_status = "Error in connection."
            with conn_lock:
                client_conn = None
            time.sleep(2)
    s.close()

# --------------------- Game Settings ---------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

# Player settings
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 30
PLAYER_SPEED = 5

# Bullet settings
BULLET_WIDTH = 5
BULLET_HEIGHT = 10
BULLET_SPEED = 7

# Enemy settings (one enemy at a time)
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_SPEED = 2

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
BLACK = (0, 0, 0)

# Global flag to control rendering mode.
# Initially, we run headless (renderEnabled = False) for faster training.
renderEnabled = True

# --------------------- Game Classes ---------------------
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.midbottom = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10)
        self.speed = 0  # Horizontal movement

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
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y -= BULLET_SPEED
        if self.rect.bottom < 0:
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
            dist = math.hypot(bullet.rect.centerx - self.enemies.sprites()[0].rect.centerx,
                            bullet.rect.centery - self.enemies.sprites()[0].rect.centery)
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

# --------------------- Main Game Function ---------------------
def main():
    global client_conn, connection_status, renderEnabled

    pygame.init()
    
    # Initialize display based on render mode
    if renderEnabled:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Shooting Game: DRL Environment")
    else:
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Create sprite groups
    all_sprites = pygame.sprite.Group()
    bullet_group = pygame.sprite.Group()
    enemy_group = pygame.sprite.Group()

    # Create player instance
    player = Player()
    all_sprites.add(player)

    current_enemy = None

    # Start network thread for DRL agent communication
    net_thread = threading.Thread(target=network_thread, daemon=True)
    net_thread.start()

    running = True
    updateCount = 0  # Count simulation updates
    while running:
        clock.tick(FPS)

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Keyboard input for testing
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player.speed = -PLAYER_SPEED
                elif event.key == pygame.K_RIGHT:
                    player.speed = PLAYER_SPEED
                elif event.key == pygame.K_SPACE:
                    # Always create new bullet when space is pressed
                    bullet = Bullet(player.rect.centerx, player.rect.top)
                    all_sprites.add(bullet)
                    bullet_group.add(bullet)
            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    player.speed = 0

        # Process network commands
        while not command_queue.empty():
            command = command_queue.get()
            if command == "LEFT":
                player.speed = -PLAYER_SPEED
            elif command == "RIGHT":
                player.speed = PLAYER_SPEED
            elif command == "STOP":
                player.speed = 0
            elif command == "SHOOT":
                # Always create new bullet when SHOOT command is received
                bullet = Bullet(player.rect.centerx, player.rect.top)
                all_sprites.add(bullet)
                bullet_group.add(bullet)

        # Ensure a single enemy exists
        if not enemy_group:
            current_enemy = Enemy()
            all_sprites.add(current_enemy)
            enemy_group.add(current_enemy)

        # Update sprites
        all_sprites.update()

        # Calculate reward
        reward = 0
        if pygame.sprite.groupcollide(bullet_group, enemy_group, True, True):
            reward += 10
        for enemy in enemy_group:
            if enemy.rect.bottom >= SCREEN_HEIGHT:
                reward -= 5
                enemy.kill()
        alignment_reward = 0
        bullet_reward = 0
        if enemy_group:
            enemy_obj = list(enemy_group)[0]
            alignment_reward = 1.0 - (abs(player.rect.centerx - enemy_obj.rect.centerx) / SCREEN_WIDTH)
            if bullet_group:
                # Calculate reward based on all bullets
                total_bullet_reward = 0
                for bullet in bullet_group:
                    dist = math.hypot(bullet.rect.centerx - enemy_obj.rect.centerx,
                                    bullet.rect.centery - enemy_obj.rect.centery)
                    max_dist = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
                    bullet_reward = 1.0 - (dist / max_dist)
                    total_bullet_reward += bullet_reward
                bullet_reward = total_bullet_reward / len(bullet_group)  # Average bullet reward
        reward += 2 * alignment_reward + 2 * bullet_reward

        # Construct and send state vector
        player_x = float(player.rect.x)
        if bullet_group:
            # Get the closest bullet to the enemy
            closest_bullet = None
            min_dist = float('inf')
            for bullet in bullet_group:
                if enemy_group:
                    enemy_obj = list(enemy_group)[0]
                    dist = math.hypot(bullet.rect.centerx - enemy_obj.rect.centerx,
                                    bullet.rect.centery - enemy_obj.rect.centery)
                    if dist < min_dist:
                        min_dist = dist
                        closest_bullet = bullet
            if closest_bullet:
                bullet_x = float(closest_bullet.rect.x)
                bullet_y = float(closest_bullet.rect.y)
            else:
                bullet_x, bullet_y = -1.0, -1.0
        else:
            bullet_x, bullet_y = -1.0, -1.0
        if enemy_group:
            enemy_obj = list(enemy_group)[0]
            enemy_x = float(enemy_obj.rect.x)
            enemy_y = float(enemy_obj.rect.y)
        else:
            enemy_x, enemy_y = -1.0, -1.0

        state_message = f"STATE:{player_x},{bullet_x},{bullet_y},{enemy_x},{enemy_y};REWARD:{reward}\n"
        with conn_lock:
            if client_conn is not None:
                try:
                    client_conn.send(state_message.encode("utf-8"))
                except Exception as e:
                    print("Error sending state:", e)
                    client_conn = None

        # Rendering
        screen.fill(BLACK)
        all_sprites.draw(screen)
        info_lines = [
            f"State: [Player X: {player_x:.1f}, Bullet: ({bullet_x:.1f}, {bullet_y:.1f}), Enemy: ({enemy_x:.1f}, {enemy_y:.1f})]",
            f"Reward: {reward:.2f}",
            f"Server: {SERVER_IP}:{TCP_PORT}",
            f"Connection: {connection_status}",
            f"Active Bullets: {len(bullet_group)}"
        ]
        for idx, line in enumerate(info_lines):
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (10, 10 + idx * 20))
        
        if renderEnabled:
            pygame.display.flip()

        # Count updates
        updateCount += 1

    pygame.quit()

if __name__ == '__main__':
    main()
