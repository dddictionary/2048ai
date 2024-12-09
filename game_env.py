import numpy as np
import torch
import random
from collections import deque
import torch.nn.functional as F
from model import DQN
import torch.optim as optim

from game_functions import (
    move_up, 
    move_down, 
    move_left, 
    move_right, 
    add_new_tile, 
    initialize_game,
    check_for_win,
    fixed_move
)

class Game2048Env:
    def __init__(self):
        self.board = None
        self.reset()
        self.moves = [move_up, move_down, move_left, move_right]

    def reset(self):
        self.board = initialize_game()
        return self.board

    def step(self, action):
        """
        Perform an action and return next state, reward, and done flag
        
        Args:
            action (int): 0: Up, 1: Down, 2: Left, 3: Right
        """
        new_board, move_made, score = self.moves[action](self.board.copy())
        
        # Reward structure
        reward = 0
        if move_made:
            # Reward for moving
            reward += score
            
            # Additional reward for creating larger tiles
            max_tile = np.max(new_board)
            reward += max_tile * 0.1
            
            # Add new tile
            new_board = add_new_tile(new_board)
        else:
            # Penalty for invalid move
            reward -= 10
        
        # Check game end conditions
        done = not move_made or check_for_win(new_board)
        
        self.board = new_board
        
        return self.board, reward, done

    def get_valid_actions(self):
        """Return list of valid actions"""
        valid_actions = []
        for i, move_func in enumerate(self.moves):
            new_board, move_made, _ = move_func(self.board.copy())
            if move_made:
                valid_actions.append(i)
        return valid_actions

class DQNAgent:
    def __init__(self, state_size=(4,4), action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(input_shape=state_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions=None):
        if valid_actions is None or len(valid_actions) == 0:
            # If no valid actions, use a fallback move
            new_board, move_made, _ = fixed_move(state)
            if move_made:
                return self.moves.index(move_made)
            return random.randint(0, self.action_size - 1)
        
        # Epsilon-greedy strategy
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state.copy()).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Filter Q-values for valid actions
        q_values = q_values.squeeze()
        valid_q_values = q_values[valid_actions]
        action_index = valid_actions[torch.argmax(valid_q_values).item()]
        
        return action_index

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state.copy()).unsqueeze(0).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state.copy()).unsqueeze(0).unsqueeze(0).to(self.device)
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))

            # Compute current Q-value
            current_q = self.model(state)[0][action]
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q, torch.tensor(target).to(self.device))
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=10000, max_steps=1000):
    env = Game2048Env()
    agent = DQNAgent()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get valid actions, handle empty case
            valid_actions = env.get_valid_actions()
            
            # If no valid actions, try a fixed move or reset
            if not valid_actions:
                new_board, move_made, _ = fixed_move(state)
                if not move_made:
                    # Game is truly stuck, break the episode
                    break
                # Use the first valid move from fixed_move
                action = env.moves.index(move_made)
            else:
                action = agent.act(state, valid_actions)
            
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        
        print(f"Episode {episode}: Total Reward = {total_reward}")
        
        # Optionally save model periodically
        if episode % 500 == 0:
            torch.save(agent.model.state_dict(), f'./snapshots/2048_dqn_model_ep{episode}.pth')

if __name__ == "__main__":
    train_agent()