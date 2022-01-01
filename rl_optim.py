import math
import random
import sys
import numpy as np
from collections import namedtuple, deque
from itertools import count
import pnp.build.pnp_python_binding

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 2, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 8, 2, 2)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 4, 2, 2)
        self.bn3 = nn.BatchNorm1d(4)

        self.head = nn.Linear(16*4, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x


# dqn = DQN(128*6).to(device)
# inp = torch.rand((1, 1, 128))
# dqn(inp)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


with open('debug/test_refine.npy', 'rb') as afile:
    xyz_array = np.load(afile)
    xy_array = np.load(afile)
print(xy_array.shape, xyz_array.shape)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()


# Get number of actions from gym action space
n_actions = xyz_array.shape[0]*xyz_array.shape[1]

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
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
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def reward_function(s, s_cost, action_, d1, d2):
    new_s = s[:]
    idx = action_ // 6
    idx2 = action_ % 6
    assert idx * 6 + idx2 == action_
    new_s[idx] = idx2
    new_cost = evaluate(new_s, d1, d2)
    reward_ = s_cost-new_cost
    terminate_ = new_cost < 100
    if reward_ < 0:
        reward_ = 0
    return new_s, new_cost, reward_, terminate_, reward_ > 0


def evaluate(s, d1, d2):
    d11 = d1[list(range(len(s))), s, :]
    d21 = d2[list(range(len(s))), s, :]
    mat = pnp.build.pnp_python_binding.pnp(d11, d21)
    d13 = np.hstack([d11, np.ones((d11.shape[0], 1))])
    xy = mat[None, :, :] @ d13[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.abs(xy - d21))/d11.shape[0]/2
    return diff


def list2tensor(_list):
    _list = np.array(_list)
    _list = _list.reshape((1, 1, -1))
    _list = _list.astype(np.float32)
    _tensor = torch.from_numpy(_list)
    return _tensor


def tensor2list(_tensor):
    assert _tensor.size(0) == 1 and _tensor.size(1) == 1
    _tensor = _tensor.view(_tensor.size(2))
    _list = [int(_tensor[_idx].item()) for _idx in range(_tensor.size(0))]
    return _list


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


actions = list(range(xyz_array.shape[1]))


state = [random.choice(actions) for _ in range(xyz_array.shape[0])]
num_episodes = 50
reward_tracks = []
best_state = None
best_cost = None

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = [random.choice(actions) for _ in range(xyz_array.shape[0])]
    current_cost = evaluate(state, xyz_array, xy_array)
    state_pt = list2tensor(state)

    for t in count():
        # Select and perform an action
        action = select_action(state_pt)
        next_state, next_cost, reward, done, better = reward_function(state, current_cost, action.item(),
                                                                      xyz_array, xy_array)

        # tracking the best state
        if best_cost is None or best_cost > current_cost:
            best_state = state
            best_cost = current_cost

        # statistics
        reward_tracks.insert(0, reward)
        if len(reward_tracks) > 1000:
            reward_tracks.pop()
        if t % 100 == 0:
            print(f"Episode {i_episode} iter {t}: reward={round(reward, 3)} "
                  f"avg. reward={round(np.mean(reward_tracks), 3)} "
                  f"best cost={round(best_cost, 3)}")

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if done:
            next_state_pt = None
        else:
            next_state_pt = list2tensor(next_state)

        # Store the transition in memory
        memory.push(state_pt, action, next_state_pt, reward)

        # Move to the next state
        if better:
            state = next_state
            state_pt = next_state_pt
            current_cost = next_cost

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')