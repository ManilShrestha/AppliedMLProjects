import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make("MountainCar-v0")
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# From OpenAI source code
def getHeight(x_position):
    return np.sin(3 * x_position) * 0.45 + 0.55

def newreward(pos):
    if (pos>=0.7):
        return 2.5
    elif (pos>=0.5):
        return 2
    else:
        return (pos+1.2)/1.8 - 1


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-3

num_episodes = 600

# Get number of actions from gym action space
n_actions = env.action_space.n

print("N_actions:",n_actions)
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


############# Show the game before trainin #################
# print("Game before training....")
# env = gym.make("MountainCar-v0", render_mode="human")

# episodes = 1

# for episode in range(1, episodes+1):
#     # Initialize the environment and get it's state
#     state, info = env.reset()
#     state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#     done = False
#     score = 0

#     while not done:
#         action = select_action(state)
#         observation, reward, terminated, truncated, _ = env.step(action.item())
#         reward = torch.tensor([reward], device=device)
#         done = terminated or truncated

#         if terminated:
#             next_state = None
#         else:
#             next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
#         # action = random.choice([0,1])
#         # _, reward, done, __ = env.step(action)
#         score += reward
#         state = next_state

#         env.render()

    
#     print(f"Environment {episode}, Score: {score}")
# env.close()

#################### Training ##########################
# env = gym.make("MountainCar-v0")

for i_episode in range(num_episodes):

    action_list =[]
    reward_list =[]
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # state = (state - env.observation_space.low) * np.array([10,50])

    # Discretize the states size
    # num_states = (env.observation_space.high - env.observation_space.low) * np.array([10,50])
    # num_states = np.round(num_states,0).astype(int) + 1

    for t in count():

        action = select_action(state)
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # discretized state
        state2_adj = np.round((observation - env.observation_space.low)*np.array([10,50]),0).astype(int)

        n_reward = newreward(observation[0])

        action_list.append(action.item())
        reward_list.append(n_reward)
        
        n_reward = torch.tensor([n_reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, n_reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        

        if done:
            episode_durations.append(np.sum(reward_list))
            # print(np.mean(np.array(reward_list)))
            print("Reward List", reward_list)
            print("Action", action_list)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

############# Show the game before trainin #################
print("Game after training....")

env = gym.make("MountainCar-v0", render_mode="human")

episodes = 3

for episode in range(1, episodes+1):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    score = 0

    while not done:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = newreward(observation[0])
        # reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        score += reward
        state = next_state

        env.render()

    
    print(f"Environment {episode}, Score: {score}")


env.close()
