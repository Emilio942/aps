import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import random

# Define the Generator Network
class PhysicsGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=10):
        super(PhysicsGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Network to generate physical parameters
        # Replacing BatchNorm with LayerNorm to handle batch size of 1
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),  # Works better with batch size of 1
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output normalized between -1 and 1
        )
        
    def forward(self, z):
        return self.model(z)

# Define the Discriminator Network
class PhysicsDiscriminator(nn.Module):
    def __init__(self, input_dim=10):
        super(PhysicsDiscriminator, self).__init__()
        
        # Network to determine if physical parameters are realistic
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Environment with Adversarial Physics
class AdversarialPhysicsEnv(gym.Env):
    def __init__(self, generator, base_gravity=9.8, base_friction=0.5, 
                 base_elasticity=0.7, difficulty_scale=0.5, max_steps=1000):
        super(AdversarialPhysicsEnv, self).__init__()
        self.generator = generator
        
        # Base physical parameters
        self.base_params = {
            'gravity': base_gravity,
            'friction': base_friction,
            'elasticity': base_elasticity,
            'wind_x': 0.0,
            'wind_y': 0.0,
            'ground_roughness': 0.0,
            'obstacle_density': 0.0,
            'terrain_variability': 0.0,
            'visibility': 1.0,
            'dynamic_obstacles': 0.0
        }
        
        # Parameters for environment
        self.difficulty_scale = difficulty_scale
        self.max_steps = max_steps
        self.current_step = 0
        
        # Simple 2D robot state: (x, y, vx, vy, orientation)
        self.state_dim = 5
        
        # Define action and observation space
        # Actions: movement in x and y directions and rotation
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5]), 
            high=np.array([1.0, 1.0, 0.5]), 
            dtype=np.float32
        )
        
        # Observations: robot state + distance sensors (8 directions)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (self.state_dim + 8)),
            high=np.array([np.inf] * (self.state_dim + 8)),
            dtype=np.float32
        )
        
        # Initialize the environment
        self.reset()
    
    def get_adversarial_params(self):
        """Generate physics parameters using the generator network"""
        z = torch.randn(1, self.generator.latent_dim)
        with torch.no_grad():
            # Set generator to eval mode to handle BatchNorm with batch size 1
            self.generator.eval()
            # Generate normalized parameters
            params = self.generator(z).numpy()[0]
            # Set back to train mode if needed
            self.generator.train()
            
        # Convert normalized params to actual physical parameters
        physics_params = self.base_params.copy()
        param_keys = list(physics_params.keys())
        
        for i, key in enumerate(param_keys):
            if i < len(params):
                # Apply difficulty scaling to make the environment progressively harder
                variation = params[i] * self.difficulty_scale
                
                # Apply different scales for different parameters
                if key == 'gravity':
                    physics_params[key] = self.base_params[key] * (1 + variation * 0.5)  # Vary gravity by ±50%
                elif key == 'friction':
                    physics_params[key] = max(0.1, min(1.0, self.base_params[key] + variation * 0.5))
                elif key == 'elasticity':
                    physics_params[key] = max(0.1, min(1.0, self.base_params[key] + variation * 0.3))
                else:
                    physics_params[key] = self.base_params[key] + variation
        
        return physics_params
    
    def reset(self, **kwargs):
        """Reset the environment for a new episode"""
        self.current_step = 0
        
        # Generate new physics parameters
        self.physics_params = self.get_adversarial_params()
        
        # Initialize robot state: (x, y, vx, vy, orientation)
        self.robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Random target position
        self.target_pos = np.array([
            np.random.uniform(-10.0, 10.0),
            np.random.uniform(-10.0, 10.0)
        ])
        
        # Generate random obstacles
        self.obstacles = []
        num_obstacles = int(10 * self.physics_params['obstacle_density'])
        for _ in range(num_obstacles):
            pos = np.array([
                np.random.uniform(-15.0, 15.0),
                np.random.uniform(-15.0, 15.0)
            ])
            radius = np.random.uniform(0.5, 2.0)
            self.obstacles.append((pos, radius))
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}  # Second element is info dict for gymnasium 0.26.0+
    
    def _get_observation(self):
        """Create observation from robot state and distance sensors"""
        # Basic state (x, y, vx, vy, orientation)
        obs = self.robot_state.copy()
        
        # Add distance sensor readings (8 directions)
        sensor_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        sensor_readings = []
        
        # Simulate distance sensors
        for angle in sensor_angles:
            # Calculate sensor direction
            direction = np.array([
                np.cos(self.robot_state[4] + angle),
                np.sin(self.robot_state[4] + angle)
            ])
            
            # Find minimum distance to obstacles in this direction
            min_distance = 10.0  # Maximum sensor range
            
            # Check distance to each obstacle
            robot_pos = self.robot_state[0:2]
            for obs_pos, obs_radius in self.obstacles:
                # Vector from robot to obstacle center
                to_obs = obs_pos - robot_pos
                
                # Project this vector onto the sensor direction
                proj_length = np.dot(to_obs, direction)
                
                # Only consider obstacles in front of the sensor
                if proj_length > 0:
                    # Find perpendicular distance to the line
                    perp_dist = np.linalg.norm(to_obs - proj_length * direction)
                    
                    # If perpendicular distance is less than obstacle radius,
                    # sensor ray intersects the obstacle
                    if perp_dist < obs_radius:
                        # Calculate distance to edge of obstacle
                        edge_dist = proj_length - np.sqrt(obs_radius**2 - perp_dist**2)
                        min_distance = min(min_distance, max(0.0, edge_dist))
            
            # Apply visibility factor (simulating fog, etc.)
            min_distance = min_distance * self.physics_params['visibility']
            sensor_readings.append(min_distance)
        
        # Add sensor readings to observation
        return np.concatenate([obs, np.array(sensor_readings)])
    
    def step(self, action):
        """Take a step in the environment given robot action"""
        self.current_step += 1
        
        # Unpack action (move_x, move_y, rotate)
        move_x, move_y, rotate = action
        
        # Update orientation
        self.robot_state[4] += rotate
        
        # Convert action to velocities based on current orientation
        orientation = self.robot_state[4]
        velocity_x = move_x * np.cos(orientation) - move_y * np.sin(orientation)
        velocity_y = move_x * np.sin(orientation) + move_y * np.cos(orientation)
        
        # Apply physics effects
        # 1. Apply wind
        velocity_x += self.physics_params['wind_x'] * 0.1
        velocity_y += self.physics_params['wind_y'] * 0.1
        
        # 2. Apply friction
        friction = self.physics_params['friction']
        velocity_x *= (1 - friction * 0.1)
        velocity_y *= (1 - friction * 0.1)
        
        # 3. Apply ground roughness (random noise)
        roughness = self.physics_params['ground_roughness']
        if roughness > 0:
            velocity_x += np.random.normal(0, roughness * 0.1)
            velocity_y += np.random.normal(0, roughness * 0.1)
        
        # Update velocities
        self.robot_state[2] = velocity_x
        self.robot_state[3] = velocity_y
        
        # Update position
        self.robot_state[0] += velocity_x
        self.robot_state[1] += velocity_y
        
        # Check for collisions with obstacles
        robot_pos = self.robot_state[0:2]
        collision = False
        
        for obs_pos, obs_radius in self.obstacles:
            distance = np.linalg.norm(robot_pos - obs_pos)
            if distance < obs_radius + 0.5:  # 0.5 is robot radius
                collision = True
                
                # Calculate bounce effect
                normal = (robot_pos - obs_pos) / distance
                
                # Apply elasticity to bounce
                elasticity = self.physics_params['elasticity']
                bounce_factor = 1.0 + elasticity
                
                # Modify velocity based on bounce
                vel = np.array([self.robot_state[2], self.robot_state[3]])
                vel_normal_component = np.dot(vel, normal)
                
                if vel_normal_component < 0:  # Only bounce if moving toward the obstacle
                    vel -= bounce_factor * vel_normal_component * normal
                    self.robot_state[2], self.robot_state[3] = vel
                    
                    # Push robot outside of obstacle
                    penetration = obs_radius + 0.5 - distance
                    self.robot_state[0] += normal[0] * penetration * 1.01
                    self.robot_state[1] += normal[1] * penetration * 1.01
        
        # Calculate reward based on distance to target
        prev_distance = np.linalg.norm(np.array([self.robot_state[0], self.robot_state[1]]) - self.target_pos)
        current_distance = np.linalg.norm(np.array([self.robot_state[0] + velocity_x, 
                                                    self.robot_state[1] + velocity_y]) - self.target_pos)
        
        # Reward for moving toward the target
        reward = (prev_distance - current_distance) * 10
        
        # Penalty for collision
        if collision:
            reward -= 5
        
        # Check if reached target
        reached_target = current_distance < 1.0
        
        # Check if episode is done
        done = reached_target or self.current_step >= self.max_steps
        
        if reached_target:
            reward += 100  # Bonus for reaching target
        
        # Get the new observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'distance_to_target': current_distance,
            'physics_params': self.physics_params,
            'reached_target': reached_target
        }
        
        # Return step results
        return observation, reward, done, False, info  # Fourth element is truncated flag for gymnasium 0.26.0+

# Define the agent that will learn to navigate in the adversarial environment
class RobotAgent:
    def __init__(self, state_dim, action_dim):
        # Define a simple neural network policy
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        
        # Discount factor
        self.gamma = 0.99
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def act(self, state, train=True):
        # Epsilon-greedy action selection
        if train and random.random() < self.epsilon:
            return np.random.uniform(-1, 1, 3)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from policy network
        with torch.no_grad():
            action = self.policy_network(state_tensor).squeeze().numpy()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        # Sample random batch from memory
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            action_tensor = torch.FloatTensor(action)
            reward_tensor = torch.FloatTensor([reward])
            next_state_tensor = torch.FloatTensor(next_state)
            
            # Get predicted action values
            predicted_action = self.policy_network(state_tensor)
            
            # Calculate target action values using Q-learning update rule
            target_action = predicted_action.clone().detach()
            
            if not done:
                # Get next action value using the policy network
                with torch.no_grad():
                    next_action_value = torch.max(self.policy_network(next_state_tensor))
                
                # Q-learning update
                target_action[0] = reward + self.gamma * next_action_value
            else:
                target_action[0] = reward
            
            # Calculate loss
            loss = nn.MSELoss()(predicted_action, target_action)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train the GAN
def train_physics_gan(generator, discriminator, real_data_func, epochs=1000, batch_size=32):
    """
    Train the GAN to generate realistic physics parameters
    
    Args:
        generator: The generator network
        discriminator: The discriminator network
        real_data_func: Function to get real physics data
        epochs: Number of training epochs
        batch_size: Batch size
    """
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    # Ensure we're in training mode
    generator.train()
    discriminator.train()
    
    # Training loop
    for epoch in range(epochs):
        # Train Discriminator
        # Real data
        real_data = real_data_func(batch_size)
        real_labels = torch.ones(batch_size, 1)
        
        # Fake data
        z = torch.randn(batch_size, generator.latent_dim)
        fake_data = generator(z).detach()
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train on real data
        discriminator_optimizer.zero_grad()
        real_outputs = discriminator(real_data)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()
        
        # Train on fake data
        fake_outputs = discriminator(fake_data)
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()
        
        discriminator_optimizer.step()
        
        # Train Generator
        generator_optimizer.zero_grad()
        z = torch.randn(batch_size, generator.latent_dim)
        fake_data = generator(z)
        outputs = discriminator(fake_data)
        
        # Generate data that tries to fool the discriminator
        generator_loss = criterion(outputs, real_labels)
        generator_loss.backward()
        generator_optimizer.step()
    
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {real_loss.item() + fake_loss.item():.4f}, G Loss: {generator_loss.item():.4f}")

# Function to get real physics data
def get_real_physics_data(batch_size):
    """
    Function to get real physics parameters from datasets or simulations
    In a real application, this would fetch actual data
    """
    # Here we're simulating "real" physics data 
    # In practice, this would come from real-world sensors or established physics simulations
    data = torch.zeros(batch_size, 10)
    
    # Generate some realistic values with constraints
    for i in range(batch_size):
        # Gravity (normalized around Earth's gravity)
        data[i, 0] = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.1))
        
        # Friction (between 0.2 and 0.8)
        data[i, 1] = torch.clamp(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.3)), -0.8, 0.8)
        
        # Elasticity (between 0.1 and 0.9)
        data[i, 2] = torch.clamp(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2)), -0.9, 0.9)
        
        # Wind forces (typically low)
        data[i, 3] = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        data[i, 4] = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        
        # Ground roughness (typically low to medium)
        data[i, 5] = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.3)))
        
        # Obstacle density (varies)
        data[i, 6] = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.4)))
        
        # Terrain variability
        data[i, 7] = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.4)))
        
        # Visibility (typically high)
        data[i, 8] = torch.clamp(torch.normal(mean=torch.tensor(0.8), std=torch.tensor(0.2)), 0.0, 1.0)
        
        # Dynamic obstacles (typically low)
        data[i, 9] = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.3)))
    
    return data

# Main training loop that combines all components
def train_adversarial_physics_system(episodes=1000, max_steps=500):
    # Initialize the GAN components
    latent_dim = 100
    physics_dim = 10
    
    generator = PhysicsGenerator(latent_dim, physics_dim)
    discriminator = PhysicsDiscriminator(physics_dim)
    
    # Pretrain the GAN
    print("Pretraining Physics GAN...")
    train_physics_gan(generator, discriminator, get_real_physics_data, epochs=500, batch_size=32)
    
    # Initialize the environment with the trained generator
    env = AdversarialPhysicsEnv(generator, difficulty_scale=0.3, max_steps=max_steps)
    
    # Initialize the robot agent
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]
    agent = RobotAgent(observation_space_size, action_space_size)
    
    # Training metrics
    rewards_history = []
    success_rate_history = []
    
    # Training loop
    print("Training Robot Agent in Adversarial Environment...")
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        reached_target = False
        
        # Episode loop
        for step in range(max_steps):
            # Select action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Check if target reached
            if info.get('reached_target', False):
                reached_target = True
            
            # Train agent
            agent.train()
            
            # Check if episode is done
            if done:
                break
        
        # Record metrics
        rewards_history.append(episode_reward)
        success_rate_history.append(1 if reached_target else 0)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            success_rate = np.mean(success_rate_history[-10:]) * 100
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            
            # Every 100 episodes, increase difficulty
            if (episode + 1) % 100 == 0 and env.difficulty_scale < 1.0:
                env.difficulty_scale += 0.1
                print(f"Increasing difficulty to {env.difficulty_scale:.1f}")
                
                # Also retrain the GAN to generate more challenging environments
                train_physics_gan(generator, discriminator, get_real_physics_data, 
                                 epochs=100, batch_size=32)
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    # Calculate moving average success rate
    window_size = 20
    moving_avg = [np.mean(success_rate_history[max(0, i-window_size):i+1]) 
                 for i in range(len(success_rate_history))]
    plt.plot(moving_avg)
    plt.title('Success Rate (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return generator, agent, env

# Example usage
if __name__ == "__main__":
    # Train the system
    generator, agent, env = train_adversarial_physics_system(episodes=500, max_steps=300)
    
    # Save the trained models
    torch.save(generator.state_dict(), "physics_generator.pth")
    torch.save(agent.policy_network.state_dict(), "robot_agent.pth")
    
    print("Training completed and models saved.")
