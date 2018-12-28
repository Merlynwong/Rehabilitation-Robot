import random
import numpy as np
import torch
import time
import os
import socket
import sys
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def get_data_array(encoded):
    dat = encoded.split('#')
    vals = []
    for temp in dat:
        if len(temp) > 0:
            try:
                vals.append(float(temp))
            except:
                pass
    return vals


host = socket.gethostname()  # get local machine name
port = 8080  # Make sure it's within the > 1024 $$ <65535 range
s = socket.socket()
s.connect((host, port))
message = ''
tar = 0


def get_feedback(x_values=[]):
    global tar
    if tar == 0:
        tar = 90
    else:
        tar = 0
    message = str(tar) + "#" + str(x_values[0]) + "#" + str(x_values[1]) + "#" + str(x_values[2])
    print("X VALUES:\n", message)
    ao = 0
    at = 0
    for i in range(5):
        s.send(message.encode('utf-8'))
        data = s.recv(1024).decode('utf-8')
        data = get_data_array(data)
        ao = ao + data[0]
        at = at + data[1]
    ao = ao/5.0
    at = at/5.0
    print("Received State Values\n", list([ao, at]))
    return list([ao, at])


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.num_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 1401
        self.replay_memory_size = 10
        self.mini_batch_size = 4
        self.x_max_values = [1000, 0.1, 300]
        self.x_min_values = [100, 0.0001, 0.1]

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def increment(x, iteration, max_range, min_range, num_iterations):
    x1_increments = torch.tensor(np.linspace(max_range[0] / 100, min_range[0] / 10, num_iterations), dtype=torch.float32)
    x2_increments = torch.tensor(np.linspace(max_range[1] / 100, min_range[1] / 10, num_iterations), dtype=torch.float32)
    x3_increments = torch.tensor(np.linspace(max_range[2] / 100, min_range[2] / 10, num_iterations), dtype=torch.float32)
    if iteration > 50:
        index = [torch.randint(iteration - 50, iteration, torch.Size([]), dtype=torch.int)][0]
    else:
        index = [torch.randint(iteration, torch.Size([]), dtype=torch.int)][0]

    if iteration < 201:
        x[0] = x[0] + x1_increments[index]  # first 200 iterations only increasing x1
    if 201 <= iteration < 401:
        x[1] = x[1] + x2_increments[index]
    if 401 <= iteration < 601:
        x[2] = x[2] + x3_increments[index]
    if 601 <= iteration < 801:
        x[0] = x[0] + x1_increments[index]
        x[1] = x[1] + x2_increments[index]
    if 801 <= iteration < 1001:
        x[1] = x[1] + x2_increments[index]
        x[2] = x[2] + x3_increments[index]
    if 1001 <= iteration < 1201:
        x[0] = x[0] + x1_increments[index]
        x[2] = x[2] + x3_increments[index]
    if 1201 <= iteration < 1401:
        x[0] = x[0] + x1_increments[index]
        x[1] = x[1] + x2_increments[index]
        x[2] = x[2] + x3_increments[index]
    return x


def decrement(x, iteration, max_range, min_range, num_iterations):
    x1_decrements = torch.tensor(np.linspace(max_range[0] / 100, min_range[0] / 10, num_iterations), dtype=torch.float32)
    x2_decrements = torch.tensor(np.linspace(max_range[1] / 100, min_range[1] / 10, num_iterations), dtype=torch.float32)
    x3_decrements = torch.tensor(np.linspace(max_range[2] / 100, min_range[2] / 10, num_iterations), dtype=torch.float32)
    if iteration > 50:
        index = [torch.randint(iteration - 50, iteration, torch.Size([]), dtype=torch.int)][0]
    else:
        index = [torch.randint(iteration, torch.Size([]), dtype=torch.int)][0]
    if iteration < 201:
        if x[0] > 0:
            x[0] = x[0] - x1_decrements[index]  # first 200 iterations only increasing x1
    if 201 <= iteration < 401:
        x[1] = x[1] - x2_decrements[index]
    if 401 <= iteration < 601:
        x[2] = x[2] - x3_decrements[index]
    if 601 <= iteration < 801:
        x[0] = x[0] - x1_decrements[index]
        x[1] = x[1] - x2_decrements[index]
    if 801 <= iteration < 1001:
        x[1] = x[1] - x2_decrements[index]
        x[2] = x[2] - x3_decrements[index]
    if 1001 <= iteration < 1201:
        x[0] = x[0] - x1_decrements[index]
        x[2] = x[2] - x3_decrements[index]
    if 1201 <= iteration < 1401:
        x[0] = x[0] - x1_decrements[index]
        x[1] = x[1] - x2_decrements[index]
        x[2] = x[2] - x3_decrements[index]
    return x


average_batch_overshoot = 0.0


def reward_calculator(prev_state, action, x, iteration):
    terminal = False
    ns = prev_state
    reward = 1
    r = 1
    global average_batch_overshoot
    average_batch_overshoot = average_batch_overshoot + prev_state.numpy()[0][1]
    if iteration % 20.0 / 1.0:
        average_batch_overshoot = 0.0
    if torch.eq(action, torch.tensor(np.array([0, 0, 1]))).all():
        state_values = torch.tensor([get_feedback(list(x.numpy()))])
        max_overshoot = torch.tensor([state_values[0][0]])
        settling_time = torch.tensor([state_values[0][1]])
        new_state = torch.cat((max_overshoot, settling_time))
        print("line 150 ps", prev_state)
        print("line 150 ns", new_state)
        if new_state.numpy()[0] == 0.0:
            if prev_state.numpy()[0][1] >= new_state()[1]:
                reward = 100
                print("Line 203")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            else:
                reward = -0.1
                print("Line 210")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] >= new_state.numpy()[0]:
            print("GR")
            if (prev_state.numpy()[0][1] - new_state.numpy()[1]) < 0.0001:
                print("TR")
                if iteration % 20 == 0:
                    if average_batch_overshoot/20.0 == 0.0:
                        reward = 10
                        terminal = True
                        print("Line 156")
                        # print("New state", new_state)
                        print("reward", reward)
                        # print("terminal", terminal)
                        # print("done")
                        ns, r = torch.unsqueeze(new_state, 0), reward
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] <= new_state.numpy()[1]:
                reward = -10
                print("Line 164")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] > new_state.numpy()[1]:
                reward = 1
                print("Line 172")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] < new_state.numpy()[0]:
            reward = -100
            print("Line 180")
            # print("New state", new_state)
            print("reward", reward)
            # print("terminal", terminal)
            # print("done")
            ns, r = torch.unsqueeze(new_state, 0), reward
    elif torch.eq(action, torch.tensor(np.array([0, 1, 0]))).all():
        x = increment(x=x, iteration=iteration, max_range=Network().x_max_values, min_range=Network().x_min_values,
                      num_iterations=Network().number_of_iterations)
        state_values = torch.tensor([get_feedback(list(x.numpy()))])
        max_overshoot = torch.tensor([state_values[0][0]])
        settling_time = torch.tensor([state_values[0][1]])
        new_state = torch.cat((max_overshoot, settling_time))
        print("line 195 ps", prev_state)
        print("line 196 ns", new_state)
        if new_state.numpy()[0] == 0.0:
            if prev_state.numpy()[0][1] >= new_state()[1]:
                reward = 100
                print("Line 268")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            else:
                reward = -0.1
                print("Line 276")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] >= new_state.numpy()[0]:
            print("GR2")
            if (prev_state.numpy()[0][1] - new_state.numpy()[1]) < 0.0001:
                print("TR2")
                if iteration % 20 == 0:
                    if average_batch_overshoot/20.0 == 0.0:
                        reward = 10
                        terminal = True
                        print("Line 199")
                        # print("New state", new_state)
                        print("reward", reward)
                        # print("terminal", terminal)
                        # print("done")
                        ns, r = torch.unsqueeze(new_state, 0), reward
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] <= new_state.numpy()[1]:
                reward = -10
                print("Line 207")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] > new_state.numpy()[1]:
                reward = 1
                print("Line 215")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] < new_state.numpy()[0]:
            reward = -100
            print("Line 223")
            # print("New state", new_state)
            print("reward", reward)
            # print("terminal", terminal)
            # print("done")
            ns, r = torch.unsqueeze(new_state, 0), reward
    elif torch.eq(action, torch.tensor(np.array([1, 0, 0]))).all():
        x = decrement(x=x, iteration=iteration, max_range=Network().x_max_values, min_range=Network().x_min_values,
                      num_iterations=Network().number_of_iterations)
        state_values = torch.tensor([get_feedback(list(x.numpy()))])
        max_overshoot = torch.tensor([state_values[0][0]])
        settling_time = torch.tensor([state_values[0][1]])
        new_state = torch.cat((max_overshoot, settling_time))
        print("line 240 ps", prev_state)
        print("line 241 ns", new_state)
        if new_state.numpy()[0] == 0.0:
            if prev_state.numpy()[0][1] > new_state()[1]:
                reward = 100
                print("Line 203")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            else:
                reward = -0.1
                print("Line 210")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] >= new_state.numpy()[0]:
            print("GR3")
            if (prev_state.numpy()[0][1] - new_state.numpy()[1]) < 0.01:
                print("TR3")
                if iteration % 20 == 0:
                    if average_batch_overshoot/20.0 == 0.0:
                        reward = 10
                        terminal = True
                        print("Line 242")
                        # print("New state", new_state)
                        print("reward", reward)
                        # print("terminal", terminal)
                        # print("done")
                        ns, r = torch.unsqueeze(new_state, 0), reward
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] <= new_state.numpy()[1]:
                reward = -10
                print("Line 250")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
            elif prev_state.numpy()[0][1] > new_state.numpy()[1]:
                reward = 1
                print("Line 258")
                # print("New state", new_state)
                print("reward", reward)
                # print("terminal", terminal)
                # print("done")
                ns, r = torch.unsqueeze(new_state, 0), reward
        elif prev_state.numpy()[0][0] < new_state.numpy()[0]:
            reward = -100
            print("Line 266")
            # print("New state", new_state)
            print("reward", reward)
            # print("terminal", terminal)
            # print("done")
            ns, r = torch.unsqueeze(new_state, 0), reward
    return ns, r, terminal


def train_model():
    start = time.time()
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # initialize replay memory
    replay_memory = []

    # initializing action to do nothing
    # [0,0,1] means no action, [0,1,0] means increase, [1,0,0] means decrease
    action = torch.tensor(np.array([0, 0, 1]))

    iteration = 1
    # initializing starting states to random values
    initial_settling_time = torch.tensor([1000])
    initial_overshoot = torch.tensor([2000])
    state = torch.cat((initial_overshoot, initial_settling_time)).unsqueeze(0)
    x1 = torch.tensor([600.0])
    # x2 = torch.tensor([0.01])
    x2 = torch.tensor([0.0])
    # x3 = torch.tensor([100.0])
    x3 = torch.tensor([0.0])
    x = torch.cat((x1, x2, x3))
    state, terminal, reward = reward_calculator(state, action, x, iteration)

    # initialize epsilon value
    epsilon = net.initial_epsilon

    epsilon_decrements = np.linspace(net.initial_epsilon, net.final_epsilon, net.number_of_iterations)

    # main infinite loop
    while iteration < net.number_of_iterations:
        # get output from the neural network
        output = net(state)[0]

        # initialize action
        action = torch.zeros([net.num_actions], dtype=torch.int)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")

        # choosing action
        action_index = [torch.randint(net.num_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action[action_index] = 1

        # get next state and reward
        # next state are overshoot and time which you'll get from PID
        # reward will be applied on x
        # print(state, action, x, iteration)
        next_state, reward, terminal = reward_calculator(state, action, x, iteration)

        # converting reward to a tensor
        reward = torch.tensor(torch.from_numpy(np.array([reward])).unsqueeze(0), dtype=torch.int)

        # save transition to replay memory
        replay_memory.append((state, action, reward, next_state, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > net.replay_memory_size:
            replay_memory.pop(0)

        while len(replay_memory) < net.mini_batch_size:
            replay_memory.append(replay_memory[-1])

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, net.mini_batch_size)

        print("terminal line 347", terminal)
        # unpack minibatch
        batch_state = torch.cat(tuple(d[0] for d in minibatch))
        batch_action = torch.cat(tuple(d[1].unsqueeze(0) for d in minibatch))
        batch_reward = torch.cat(tuple(d[2] for d in minibatch))
        batch_next_state = torch.cat(tuple(d[3] for d in minibatch))

        # print("batch reward", batch_reward)
        # get output for the next state
        output_next_state = net(batch_next_state)

        # set next_Q to reward_iter for terminal state, otherwise to reward_iter + gamma*max(Q)
        next_q = torch.cat(tuple(batch_reward[i]
                                 if minibatch[i][4] else batch_reward[i] + net.gamma *
                                                         torch.tensor(torch.max(output_next_state[i]), dtype=torch.int)
                                 for i in range(len(minibatch))))

        # extract present Q-value
        q_value = torch.sum(net(batch_state) * torch.tensor(batch_action, dtype=torch.float32), dim=1)

        # Resetting gradients
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        next_q = next_q.detach()

        # calculate loss
        loss = criterion(q_value, torch.tensor(next_q, dtype=torch.float32))

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state = next_state
        state = next_state
        iteration += 1
        if iteration % 10 == 0:
            torch.save(net, "current_model.pth")
        if terminal:
            torch.save(net, "current_model.pth")
            return

        # print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
        #             action.detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
        #             np.max(output.detach().numpy()))
        # print("Neural Network Diagnostics\n\nNext State Input\t\t\t\tAction Output\t\t\t\tLoss\n", batch_next_state,
        #       "\t\t\t\t", output_next_state, "\t\t\t\t", loss, "\n")


def test(net):
    # [0,0,1] means no action, [0,1,0] means increase, [1,0,0] means decrease
    action = torch.tensor(np.array([0, 0, 1]))

    iteration = 0
    # initializing starting states to random values
    initial_settling_time = 1000
    initial_overshoot = 2000
    state = torch.cat((initial_overshoot, initial_settling_time))
    x1 = torch.randint(100, 1000, torch.Size([1]))
    x2 = torch.rand(torch.Size([1])) / 10.0
    x3 = torch.rand(torch.Size([1])) * 300
    x = torch.cat((x1, x2, x3))
    state, terminal, reward = reward_calculator(state, action, x, iteration)

    while terminal is False:
        # get output from the neural network
        output = net(state)[0]
        action = torch.zeros([net.num_actions], dtype=torch.int)

        # get action
        action_index = torch.argmax(output)
        action[action_index] = 1

        # get next state
        next_state, reward, terminal = reward_calculator(state, action, x, iteration)

        # set state to be next_state
        state = next_state
    # print("Minimised X is as follows...\nx1 \t\t x2 \t\t x3 \n", x[0], "\t\t", x[1], "\t\t", x[2])


def main():
    model = Network()
    net_total_params = sum(p.numel() for p in model.parameters())
    print(net_total_params)
    train_model()
    net = torch.load('current_model.pth').eval()
    test(net)


if __name__ == "__main__":
    main()
s.close()
