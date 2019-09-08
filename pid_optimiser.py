'''
Q Learning Based PID Optimizer
Framework: PyTorch
Author: Satrajit Chatterjee, Prabin Rath
'''
from comet_ml import Experiment as comex
import random
import numpy as np
import torch
import time
import math
import socket
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

experiment = comex(project_name="pid_optimizer", api_key="s1p5x7LK2xxryou4PJ2D7kwvw")

#Decoding the recieved data
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
port = 1234  # Make sure it's within the > 1024 $$ <65535 range
s = socket.socket()
s.connect((host, port))
message = ''
tar = 0

#Function to sample the overshoot and settling time for getting feedack from the motor
def get_feedback(x_values=[]):
    global tar
    if tar == 0:
        tar = 180
    else:
        tar = 0
    message = str(tar) + "#" + str(x_values[0]) + "#" + str(x_values[1]) + "#" + str(x_values[2])
    print("X VALUES:\n", message)
    ao = 0
    at = 0
    for i in range(5):
        s.send(message.encode('utf-8'))
        time.sleep(2)
        data = s.recv(1024).decode('utf-8')
        data = get_data_array(data)
        ao = ao + data[0]
        at = at + data[1]
    print("Received State Values\n", list([ao, at]))
    return list([ao, at])

#Q Learning Implementation
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.num_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 1401
        self.replay_memory_size = 10
        self.mini_batch_size = 8
        self.x_max_values = [1000, 0.1, 300]
        self.x_min_values = [100, 0.0001, 0.1]

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x), 1)
        print(x[0], x[1], x[2])
        # exit(0)
        # if x[0] > 300.0:
        #     x[0] = 300.0
        # if x[0] < 100.0:
        #     x[0] = 100.0
        # if x[1] > 1.0:
        #     x[1] = 1.0
        # if x[1] < 0.0:
        #     x[1] = 0.0
        # if x[2] > 300.0:
        #     x[2] = 300.0
        # if x[2] < 0.0:
        #     x[2] = 0.0
        return x
        


average_batch_overshoot = 0.0
reward_zero_counter = 0
update_flag = True
previous_reward = 0
prev_time = -1
change_reward = False
previous_time = 0.0
termination_counter = 0

#To prevent the network to explore the undefind K value space
def squeeze(x):
    if x[0] > 300.0:
        x[0] = 300.0
    if x[0] < 100.0:
        x[0] = 100.0
    if x[1] > 1.0:
        x[1] = 1.0
    if x[1] < 0.0:
        x[1] = 0.0
    if x[2] > 300.0:
        x[2] = 300.0
    if x[2] < 0.0:
        x[2] = 0.0
    return x


global_slope = 0
#Reward function for reinforcement learning
def reward_calculator(action, x, iteration):
    global average_batch_overshoot, reward_zero_counter, change_reward, update_flag, previous_reward, prev_time, \
        previous_time, termination_counter, global_slope

    overshoot_slope = -global_slope
    settling_time_slope = global_slope
    to_return = []
    terminal = False
    if update_flag:
        print("line 109 action", action)
        x[0] = x[0] + action[0]
        x[1] = x[1] + action[1]
        x[2] = x[2] + action[2]
        x = squeeze(x)  # TODO: cannot squeeze like this as not a learnable operation
    state_values = torch.tensor([get_feedback(list(x.detach().numpy()))])
    max_overshoot = torch.tensor([state_values[0][0]])
    settling_time = torch.tensor([state_values[0][1]])
    new_state = torch.cat((max_overshoot, settling_time))
    average_batch_overshoot = average_batch_overshoot + new_state[0]
    if iteration % 5 == 0:
        if average_batch_overshoot / 5.0 == 0.0:
            change_reward = True
            print("Reward function changed!")
        average_batch_overshoot = 0.0
    if not change_reward:
        print("Trying to fix overshoot")
        to_return = [(new_state[0]) * -global_slope, new_state]
    else:
        if new_state[0] > 0.0:
            print("Overshoot happened really bad")
            to_return = [overshoot_slope * (new_state[0]), new_state]
        else:
            print("Trying to get the optimal value")
            if previous_time - new_state[1] < 0.002 and change_reward is True:
                termination_counter += 1
            else:
                termination_counter = 0
            if termination_counter == 20:
                terminal = True
            # to_return = [settling_time_slope * (1 / new_state[1]), new_state]
            to_return = [settling_time_slope * (previous_time - new_state[1]), new_state]
    if to_return[0] == 0:
        if previous_reward == 0:
            if prev_time < 0:
                prev_time = new_state[1]
            elif new_state[1] - prev_time < 0.2:
                update_flag = True
            else:
                print("Cannot change values further. Otherwise it will stop.")
                update_flag = False
        else:
            prev_time = -1
    previous_reward = to_return[0]
    previous_time = new_state[1]
    print("Current reward: ", to_return[0])
    return to_return[0], to_return[1], terminal


def train_model():
    start = time.time()
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # initialize replay memory
    replay_memory = []

    # initializing action
    action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    iteration = 1
    # initializing starting states to random values
    initial_settling_time = torch.tensor([1000])
    initial_overshoot = torch.tensor([2000])
    state = torch.cat((initial_overshoot, initial_settling_time)).unsqueeze(0)
    x1 = torch.tensor([random.uniform(0, 300)]) #when i put a 100 here at lower range the values get stuck. I need the kp to be high initially such that the overshoot is fixed by the algorithm. with out high kp the initial lowervalues gives 0 overshoot and the algorithm is not able to optimize the settling time by increasing kp
    # x2 = torch.tensor([0.01])
    x2 = torch.tensor([random.uniform(0, 1)])
    # x3 = torch.tensor([100.0])
    x3 = torch.tensor([random.uniform(0, 200)])
    x = torch.cat((x1, x2, x3))
    reward, state, terminal = reward_calculator(action, x, iteration)

    # initialize epsilon value
    epsilon = net.initial_epsilon

    epsilon_decrements = np.linspace(net.initial_epsilon, net.final_epsilon, net.number_of_iterations)

    # main infinite loop
    prev_loss = 0
    end_optimal = 0
    while iteration < net.number_of_iterations:
        '''
        logging: reward, overshoot, settling time, loss
        '''
        
        # get output from the neural network
        experiment.log_metric("overshoot", state[0], step=iteration)
        experiment.log_metric("settling_time", state[1], step=iteration)
        output = net(state)

        # initialize action
        action = torch.zeros([net.num_actions], dtype=torch.float32)

        # epsilon greedy exploration
        # random_action = random.random() <= epsilon
        random_action = False
        if random_action:
            print("Performed random action!")

        # choosing action
        if random_action:
            action = torch.randn(net.num_actions)
        else:
            action = output

        # get next state and reward
        # next state are overshoot and time which you'll get from PID
        # print(state, action, x, iteration)
        reward, next_state, terminal = reward_calculator(action, x, iteration)
        experiment.log_metric("reward", reward, step=iteration)

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

        print("terminal ", terminal)
        # unpack minibatch
        batch_state = torch.cat(tuple(d[0].unsqueeze(0) for d in minibatch))
        # print("line 217", tuple(d[1].size() for d in minibatch))
        batch_action = torch.cat(tuple(d[1].unsqueeze(0) for d in minibatch))
        batch_reward = torch.cat(tuple(d[2] for d in minibatch))
        batch_next_state = torch.cat(tuple(d[3].unsqueeze(0) for d in minibatch))

        # print("batch reward", batch_reward)
        # get output for the next state
        # print(batch_next_state)
        output_next_state = net(batch_next_state)

        # set next_Q to reward_iter for terminal state, otherwise to reward_iter + gamma*max(Q)
        next_q = torch.cat(tuple(batch_reward[i]
                                 if minibatch[i][4] else batch_reward[i] + torch.tensor(net.gamma *
                                                                                        torch.sum(output_next_state),
                                                                                        dtype=torch.int)
                                 for i in range(len(minibatch))))
        # print("next q", next_q)
        # print("index 0", torch.tensor(next_q[0], dtype=torch.float32))
        # print("full sum", torch.sum(next_q, dtype=torch.float32))
        # print("next_q_1", next_q)
        # extract present Q-value
        # print("line 234", batch_action)
        # print("net(batch_state)", net(batch_state))
        # print("batch_action", batch_action)
        q_value = torch.sum(net(batch_state) * batch_action).unsqueeze(0)
        # print("q value", q_value)

        # Resetting gradients
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        # next_q_ = next_q.detach()

        # calculate loss
        # print("q_value", q_value)
        # print("next_q", torch.sum(next_q))
        # loss = criterion(q_value, torch.tensor(next_q[0], dtype=torch.float32))
        loss = criterion(q_value, torch.sum(next_q, dtype=torch.float32)) * 10e-5
        print("LOSS: ", loss)
        experiment.log_metric("loss", loss.item(), step=iteration)
        # do backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

        # setting slopes
        global global_slope
        global_slope = (loss.item() - prev_loss) ** 2
        prev_loss = loss.item()

        # set state = next_state
        state = next_state
        if state[0] == 0.0 and 1 < abs(math.log(abs(reward.item()+0.01), 10)) < 3:
            end_optimal += 1
        else:
            end_optimal-=10
            if end_optimal<0:
                end_optimal = 0;
        iteration += 1
        if iteration % 10 == 0:
            torch.save(net, "current_model.pth")
        print("end_optimal", end_optimal)
        if end_optimal == 25:
            import winsound
            frequency = 500  # Set Frequency To 2500 Hertz
            duration = 1000  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
            print("Training complete!")
            print("Optimal values:\n", x)
            torch.save(net, "current_model.pth")
            return
        print("iteration", iteration)
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
        output, _ = net(state)[0]
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
    # test(net)


if __name__ == "__main__":
    main()
s.close()
