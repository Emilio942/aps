import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

# Konstanten
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator Network with Spektrales Normalisierung für stabilere GAN-Training
class PhysicsGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=10):
        super(PhysicsGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Verbesserte Architektur mit Residual-Verbindungen
        self.fc1 = nn.Linear(latent_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, output_dim)
        
    def forward(self, z):
        h1 = F.leaky_relu(self.ln1(self.fc1(z)), 0.2)
        h2 = F.leaky_relu(self.ln2(self.fc2(h1)), 0.2)
        h3 = F.leaky_relu(self.ln3(self.fc3(h2)), 0.2)
        # Residual-Verbindung
        h3 = h3 + h1
        x = torch.tanh(self.fc4(h3))
        return x

# Verbesserte Discriminator mit Gradient Penalty für WGAN-GP-Ansatz
class PhysicsDiscriminator(nn.Module):
    def __init__(self, input_dim=10):
        super(PhysicsDiscriminator, self).__init__()
        
        # Tieferes Netzwerk mit Dropout für bessere Generalisierung
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Environment with Adversarial Physics
class AdversarialPhysicsEnv(gym.Env):
    def __init__(self, generator, base_gravity=9.8, base_friction=0.5, 
                 base_elasticity=0.7, difficulty_scale=0.5, max_steps=1000,
                 render_mode=None):
        super(AdversarialPhysicsEnv, self).__init__()
        self.generator = generator
        self.render_mode = render_mode
        
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
        
        # Observations: robot state + distance sensors (8 directions) + physics_params (10)
        # Hinzufügen der Physikparameter zur Beobachtung für bessere Anpassungsfähigkeit
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (self.state_dim + 8 + 10)),
            high=np.array([np.inf] * (self.state_dim + 8 + 10)),
            dtype=np.float32
        )
        
        # Für Visualisierung
        if render_mode == 'human':
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
        
        # Initialize the environment
        self.reset()
    
    def get_adversarial_params(self):
        """Generate physics parameters using the generator network"""
        z = torch.randn(1, self.generator.latent_dim, device=DEVICE)
        with torch.no_grad():
            # Set generator to eval mode to handle BatchNorm with batch size 1
            self.generator.eval()
            # Generate normalized parameters
            params = self.generator(z).cpu().numpy()[0]
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
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
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
        
        # Dynamic obstacles
        self.dynamic_obstacles = []
        num_dynamic = int(5 * self.physics_params['dynamic_obstacles'])
        for _ in range(num_dynamic):
            pos = np.array([
                np.random.uniform(-15.0, 15.0),
                np.random.uniform(-15.0, 15.0)
            ])
            velocity = np.array([
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2)
            ])
            radius = np.random.uniform(0.5, 1.5)
            self.dynamic_obstacles.append((pos, velocity, radius))
        
        # Get initial observation
        observation = self._get_observation()
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
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
            
            # Check dynamic obstacles too
            for obs_pos, _, obs_radius in self.dynamic_obstacles:
                to_obs = obs_pos - robot_pos
                proj_length = np.dot(to_obs, direction)
                
                if proj_length > 0:
                    perp_dist = np.linalg.norm(to_obs - proj_length * direction)
                    if perp_dist < obs_radius:
                        edge_dist = proj_length - np.sqrt(obs_radius**2 - perp_dist**2)
                        min_distance = min(min_distance, max(0.0, edge_dist))
            
            # Apply visibility factor (simulating fog, etc.)
            min_distance = min_distance * self.physics_params['visibility']
            sensor_readings.append(min_distance)
        
        # Add physics parameters to observation
        physics_values = np.array(list(self.physics_params.values()))
        
        # Add sensor readings to observation
        return np.concatenate([obs, np.array(sensor_readings), physics_values])
    
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
        
        # Temporary position for collision detection
        temp_pos_x = self.robot_state[0] + velocity_x
        temp_pos_y = self.robot_state[1] + velocity_y
        
        # Update velocities
        self.robot_state[2] = velocity_x
        self.robot_state[3] = velocity_y
        
        # Check for collisions with obstacles
        robot_pos = np.array([temp_pos_x, temp_pos_y])
        collision = False
        
        # Calculate distance to target before moving
        prev_distance = np.linalg.norm(self.robot_state[0:2] - self.target_pos)
        
        # Move dynamic obstacles
        for i, (obs_pos, obs_vel, obs_radius) in enumerate(self.dynamic_obstacles):
            # Update position
            new_pos = obs_pos + obs_vel
            
            # Simple boundary reflection
            if abs(new_pos[0]) > 15.0:
                obs_vel[0] *= -1
            if abs(new_pos[1]) > 15.0:
                obs_vel[1] *= -1
                
            new_pos = obs_pos + obs_vel
            self.dynamic_obstacles[i] = (new_pos, obs_vel, obs_radius)
        
        # Check for collisions with static obstacles
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
                    temp_pos_x += normal[0] * penetration * 1.01
                    temp_pos_y += normal[1] * penetration * 1.01
        
        # Check for collisions with dynamic obstacles
        for obs_pos, _, obs_radius in self.dynamic_obstacles:
            distance = np.linalg.norm(robot_pos - obs_pos)
            if distance < obs_radius + 0.5:
                collision = True
                
                # Similar collision response as with static obstacles
                normal = (robot_pos - obs_pos) / distance
                elasticity = self.physics_params['elasticity']
                bounce_factor = 1.0 + elasticity
                
                vel = np.array([self.robot_state[2], self.robot_state[3]])
                vel_normal_component = np.dot(vel, normal)
                
                if vel_normal_component < 0:
                    vel -= bounce_factor * vel_normal_component * normal
                    self.robot_state[2], self.robot_state[3] = vel
                    
                    penetration = obs_radius + 0.5 - distance
                    temp_pos_x += normal[0] * penetration * 1.01
                    temp_pos_y += normal[1] * penetration * 1.01
        
        # Update position after handling all collisions
        self.robot_state[0] = temp_pos_x
        self.robot_state[1] = temp_pos_y
        
        # Calculate current distance to target
        current_distance = np.linalg.norm(np.array([self.robot_state[0], self.robot_state[1]]) - self.target_pos)
        
        # Reward for moving toward the target
        distance_reward = (prev_distance - current_distance) * 10
        
        # Energy efficiency reward (punish excessive actions)
        energy_penalty = -0.01 * np.sum(np.square(action))
        
        # Final reward
        reward = distance_reward + energy_penalty
        
        # Penalty for collision
        if collision:
            reward -= 5
        
        # Check if reached target
        reached_target = current_distance < 1.0
        
        # Bonus for reaching target
        if reached_target:
            reward += 100
        
        # Check if episode is done
        done = reached_target or self.current_step >= self.max_steps
        
        # Get the new observation
        observation = self._get_observation()
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        # Create info dictionary
        info = {
            'distance_to_target': current_distance,
            'physics_params': self.physics_params,
            'reached_target': reached_target,
            'collision': collision,
            'steps': self.current_step
        }
        
        # Return step results (gymnasium 0.26+ format)
        return observation, reward, done, False, info  # Fourth element is truncated flag
    
    def render(self):
        """Render the environment"""
        if self.render_mode != 'human':
            return
            
        self.ax.clear()
        
        # Set bounds
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        
        # Draw robot
        robot_circle = plt.Circle((self.robot_state[0], self.robot_state[1]), 0.5, color='blue')
        self.ax.add_patch(robot_circle)
        
        # Draw robot orientation
        orientation = self.robot_state[4]
        direction_x = 0.5 * np.cos(orientation)
        direction_y = 0.5 * np.sin(orientation)
        self.ax.arrow(self.robot_state[0], self.robot_state[1], 
                      direction_x, direction_y, head_width=0.3, color='black')
        
        # Draw target
        target_circle = plt.Circle((self.target_pos[0], self.target_pos[1]), 0.5, color='green')
        self.ax.add_patch(target_circle)
        
        # Draw obstacles
        for obs_pos, obs_radius in self.obstacles:
            obstacle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='red', alpha=0.7)
            self.ax.add_patch(obstacle)
            
        # Draw dynamic obstacles
        for obs_pos, _, obs_radius in self.dynamic_obstacles:
            obstacle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='purple', alpha=0.7)
            self.ax.add_patch(obstacle)
        
        # Draw sensor rays
        sensor_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        robot_pos = self.robot_state[0:2]
        
        for angle in sensor_angles:
            direction = np.array([
                np.cos(self.robot_state[4] + angle),
                np.sin(self.robot_state[4] + angle)
            ])
            
            # Draw sensor ray up to 10 units
            self.ax.plot([robot_pos[0], robot_pos[0] + direction[0] * 10],
                         [robot_pos[1], robot_pos[1] + direction[1] * 10], 
                         'y-', alpha=0.3)
        
        # Draw environment information
        info_text = f"Steps: {self.current_step}\n"
        info_text += f"Gravity: {self.physics_params['gravity']:.2f}\n"
        info_text += f"Friction: {self.physics_params['friction']:.2f}\n"
        info_text += f"Wind: ({self.physics_params['wind_x']:.2f}, {self.physics_params['wind_y']:.2f})"
        
        self.ax.text(-19, 18, info_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        if self.render_mode == 'human':
            plt.close()

# Verbesserer Agent mit DDPG (Deep Deterministic Policy Gradient)
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_high, hidden_dim=256, gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-3):
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Target network update rate
        self.action_high = action_high
        
        # Actor network & target
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output between -1 and 1
        ).to(DEVICE)
        
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(DEVICE)
        
        # Hard copy of parameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic network & target (Q-value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(DEVICE)
        
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(DEVICE)
        
        # Hard copy of parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.memory = deque(maxlen=100000)
        
        # Noise process for exploration
        self.noise_scale = 0.1
        
        # Training parameters
        self.batch_size = 64
        self.update_every = 4
        self.update_counter = 0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, add_noise=True):
        """Select an action given current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze()
        self.actor.train()
        
        # Add exploration noise
        if add_noise:
            action += np.random.normal(0, self.noise_scale, size=action.shape)
            
        # Clip action to valid range
        return np.clip(action, -1.0, 1.0)
    
    def learn(self):
        """Update policy and value parameters using batch of experience tuples"""
        if len(self.memory) < self.batch_size:
            return
        
        # Increment update counter
        self.update_counter += 1
        if self.update_counter % self.update_every != 0:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(DEVICE)
        actions = torch.FloatTensor([experience[1] for experience in minibatch]).to(DEVICE)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(DEVICE)
        dones = torch.FloatTensor([float(experience[4]) for experience in minibatch]).unsqueeze(1).to(DEVICE)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Reduce noise over time
        self.noise_scale = max(0.02, self.noise_scale * 0.999)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Verbesserte GAN-Training-Funktion mit Wasserstein-Loss und Gradient-Penalty
def train_physics_gan(generator, discriminator, real_data_func, epochs=1000, batch_size=32, 
                      gradient_penalty_weight=10.0, n_critic=5):
    """
    Train the GAN to generate realistic physics parameters using WGAN-GP approach
    
    Args:
        generator: The generator network
        discriminator: The discriminator network
        real_data_func: Function to get real physics data
        epochs: Number of training epochs
        batch_size: Batch size
        gradient_penalty_weight: Weight for gradient penalty
        n_critic: Number of critic updates per generator update
    """
    generator.to(DEVICE)
    discriminator.to(DEVICE)
    
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    
    # LR schedulers
    generator_scheduler = optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.99)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.99)
    
    # TensorBoard writer
    writer = SummaryWriter()
    
    # Ensure we're in training mode
    generator.train()
    discriminator.train()
    
    # Training loop
    for epoch in range(epochs):
        for _ in range(n_critic):
            # Train Discriminator
            discriminator.zero_grad()
            
            # Real data
            real_data = real_data_func(batch_size).to(DEVICE)
            
            # Fake data
            z = torch.randn(batch_size, generator.latent_dim, device=DEVICE)
            fake_data = generator(z).detach()
            
            # Compute discriminator outputs
            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_data)
            
            # Compute Wasserstein loss
            disc_loss = fake_validity.mean() - real_validity.mean()
            
            # Compute gradient penalty
            alpha = torch.rand(batch_size, 1, device=DEVICE)
            # Get random interpolations between real and fake samples
            interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
            d_interpolates = discriminator(interpolates)
            
            # Get gradients w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates, device=DEVICE),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Compute gradient penalty
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = gradient_penalty_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # Add gradient penalty to discriminator loss
            disc_loss = disc_loss + gradient_penalty
            
            # Backpropagation
            disc_loss.backward()
            discriminator_optimizer.step()
        
        # Train Generator
        generator.zero_grad()
        
        # Generate fake samples
        z = torch.randn(batch_size, generator.latent_dim, device=DEVICE)
        fake_data = generator(z)
        fake_validity = discriminator(fake_data)
        
        # Generator wants discriminator to predict real for fake samples
        gen_loss = -fake_validity.mean()
        
        # Backpropagation
        gen_loss.backward()
        generator_optimizer.step()
        
        # Update learning rates
        if (epoch + 1) % 100 == 0:
            generator_scheduler.step()
            discriminator_scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Generator', gen_loss.item(), epoch)
        writer.add_scalar('Loss/Discriminator', disc_loss.item(), epoch)
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")
    
    writer.close()
    return generator, discriminator

# Verbesserte Funktion für realistische Physikdaten
def get_real_physics_data(batch_size):
    """
    Function to get real physics parameters with realistic correlations and constraints
    """
    data = torch.zeros(batch_size, 10)
    
    for i in range(batch_size):
        # Start with base Earth-like conditions
        gravity_var = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.1))
        
        # Friction is correlated with ground roughness
        base_roughness = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.3)))
        friction_var = 0.7 * base_roughness + 0.3 * torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        friction_var = torch.clamp(friction_var, -0.8, 0.8)
        
        # Elasticity is inversely related to friction
        elasticity_var = -0.5 * friction_var + 0.5 * torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        elasticity_var = torch.clamp(elasticity_var, -0.9, 0.9)
        
        # Wind forces are correlated
        wind_strength = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.3))
        wind_direction = torch.rand(1) * 2 * torch.pi
        wind_x = wind_strength * torch.cos(wind_direction)
        wind_y = wind_strength * torch.sin(wind_direction)
        
        # Terrain variability is correlated with roughness
        terrain_var = 0.8 * base_roughness + 0.2 * torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        
        # Visibility is inversely related to wind strength (dust, fog effect)
        visibility = 1.0 - 0.6 * torch.abs(wind_strength) + 0.4 * torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        visibility = torch.clamp(visibility, 0.0, 1.0)
        
        # Obstacle and dynamic obstacle density are related
        obstacle_base = torch.abs(torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.4)))
        obstacle_density = obstacle_base
        dynamic_obstacles = 0.7 * obstacle_base + 0.3 * torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.2))
        dynamic_obstacles = torch.clamp(dynamic_obstacles, 0.0, 1.0)
        
        # Combine all parameters
        data[i, 0] = gravity_var
        data[i, 1] = friction_var
        data[i, 2] = elasticity_var
        data[i, 3] = wind_x
        data[i, 4] = wind_y
        data[i, 5] = base_roughness
        data[i, 6] = obstacle_density
        data[i, 7] = terrain_var
        data[i, 8] = visibility
        data[i, 9] = dynamic_obstacles
    
    return data

# Verbesserte Haupttrainingsfunktion
def train_adversarial_physics_system(episodes=1000, max_steps=500, 
                                    eval_frequency=50, render=False, 
                                    save_path='models'):
    """
    Main training function for the complete adversarial physics system
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_frequency: How often to evaluate the agent
        render: Whether to render the environment during evaluation
        save_path: Path to save trained models
    """
    # Initialize the GAN components
    latent_dim = 100
    physics_dim = 10
    
    generator = PhysicsGenerator(latent_dim, physics_dim)
    discriminator = PhysicsDiscriminator(physics_dim)
    
    # Create TensorBoard writer
    writer = SummaryWriter()
    
    # Pretrain the GAN
    print("Pretraining Physics GAN...")
    generator, discriminator = train_physics_gan(generator, discriminator, get_real_physics_data, 
                                               epochs=500, batch_size=32)
    
    # Initialize the environment with the trained generator
    env = AdversarialPhysicsEnv(generator, difficulty_scale=0.3, max_steps=max_steps)
    
    # Initialize the DDPG agent
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    agent = DDPGAgent(observation_space_size, action_space_size, action_high)
    
    # Training metrics
    rewards_history = []
    success_rate_history = []
    
    # Progress bar
    pbar = tqdm(total=episodes)
    
    # Training loop
    print("Training Robot Agent in Adversarial Environment...")
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        reached_target = False
        actor_loss = 0
        critic_loss = 0
        
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
            losses = agent.learn()
            if losses is not None:
                critic_loss, actor_loss = losses
            
            # Check if episode is done
            if done:
                break
        
        # Record metrics
        rewards_history.append(episode_reward)
        success_rate_history.append(1 if reached_target else 0)
        
        # Log to TensorBoard
        writer.add_scalar('Training/Reward', episode_reward, episode)
        writer.add_scalar('Training/SuccessRate', 1 if reached_target else 0, episode)
        writer.add_scalar('Training/Steps', step + 1, episode)
        writer.add_scalar('Loss/Actor', actor_loss, episode)
        writer.add_scalar('Loss/Critic', critic_loss, episode)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_description(f"Reward: {episode_reward:.2f}, Steps: {step+1}")
        
        # Evaluate periodically
        if (episode + 1) % eval_frequency == 0:
            avg_reward, success_rate = evaluate_agent(agent, env, num_episodes=10, render=render and episode % 100 == 0)
            writer.add_scalar('Evaluation/AvgReward', avg_reward, episode)
            writer.add_scalar('Evaluation/SuccessRate', success_rate, episode)
            
            print(f"\nEpisode {episode+1}/{episodes}, Eval Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            
            # Every 100 episodes, increase difficulty and retrain GAN
            if (episode + 1) % 100 == 0 and env.difficulty_scale < 1.0:
                old_difficulty = env.difficulty_scale
                env.difficulty_scale += 0.1
                print(f"Increasing difficulty from {old_difficulty:.1f} to {env.difficulty_scale:.1f}")
                
                # Retrain the GAN to generate more challenging environments
                print("Retraining Physics GAN...")
                generator, discriminator = train_physics_gan(generator, discriminator, get_real_physics_data, 
                                                          epochs=100, batch_size=32)
                env.generator = generator
    
    pbar.close()
    writer.close()
    
    # Save the final models
    os.makedirs(save_path, exist_ok=True)
    torch.save(generator.state_dict(), f"{save_path}/physics_generator.pth")
    torch.save(discriminator.state_dict(), f"{save_path}/physics_discriminator.pth")
    torch.save(agent.actor.state_dict(), f"{save_path}/robot_agent_actor.pth")
    torch.save(agent.critic.state_dict(), f"{save_path}/robot_agent_critic.pth")
    
    print(f"Training completed and models saved to {save_path}/")
    
    return generator, agent, env

# Evaluation function
def evaluate_agent(agent, env, num_episodes=10, render=False):
    """Evaluate the agent's performance without exploration noise"""
    total_rewards = []
    successes = 0
    
    # Create a separate environment for evaluation with rendering if needed
    eval_env = AdversarialPhysicsEnv(
        env.generator, 
        difficulty_scale=env.difficulty_scale, 
        max_steps=env.max_steps,
        render_mode='human' if render else None
    )
    
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, add_noise=False)  # No exploration during evaluation
            next_state, reward, done, _, info = eval_env.step(action)
            state = next_state
            episode_reward += reward
            
            if info.get('reached_target', False):
                successes += 1
                
        total_rewards.append(episode_reward)
    
    # Close rendering
    if render:
        eval_env.close()
        
    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    success_rate = (successes / num_episodes) * 100
    
    return avg_reward, success_rate

# Beispiel für die Verwendung
if __name__ == "__main__":
    import os
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Train the system
    print(f"Training on device: {DEVICE}")
    generator, agent, env = train_adversarial_physics_system(
        episodes=500, 
        max_steps=300,
        render=True,
        save_path="results"
    )
    
    # Final evaluation with rendering
    print("\nFinal evaluation with rendering...")
    eval_env = AdversarialPhysicsEnv(
        generator, 
        difficulty_scale=env.difficulty_scale, 
        max_steps=1000,
        render_mode='human'
    )
    
    # Run a few episodes with the trained agent
    for i in range(5):
        state, _ = eval_env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"\nStarting evaluation episode {i+1}/5")
        time.sleep(1)  # Pause to observe
        
        while not done:
            action = agent.act(state, add_noise=False)
            next_state, reward, done, _, info = eval_env.step(action)
            state = next_state
            total_reward += reward
            step += 1
            
            # Small delay to better visualize the agent's behavior
            time.sleep(0.05)
            
            if done:
                status = "Success!" if info.get('reached_target', False) else "Failed"
                print(f"Episode finished after {step} steps. {status} Total reward: {total_reward:.2f}")
                time.sleep(1)  # Pause to observe final state
    
    eval_env.close()
    print("Evaluation completed.")
