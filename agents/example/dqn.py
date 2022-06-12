import numpy as np
import torch
from torch import optim
import _pickle

from src.Model.Agent.agent_base import AgentBase as AgentBase

from agents.example.nn import NN
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Agent as agent_config
import torch.nn as nn

""" Agent implementation that will take the last key as an input. Do not sent any object 
to the device in __init__ as the value will be lost when process starts"""


class Agent(AgentBase):
    def __init__(self, in_queue, out_queue):
        super().__init__(in_queue, out_queue)

        self.main_model = NN().to(self.device)
        self.target_model = NN().eval().to(self.device)

        self.optimizer = optim.Adam(self.main_model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

        self.last_action = np.zeros(self.key_output_size, dtype=np.float32)

        self.counter = 0

    def get_data(self):
        """
        Override parent method to include the last key
        """
        data = self._input_queue.get()
        state_with_actions = (*data['state'][:3], self.last_action)
        data['state'] = state_with_actions

        return data

    def action(self, action=None):
        """
        Override parent method to include the last key
        """
        # if action is not None:
        #     self._action_out_queue.put(action)

        data = self.get_data()
        self.last_action = action

        return data['state'], data['reward']

    def train(self):
        batch = self.replay_memory.sample()

        if len(batch) < nn_config.get_batch_size():
            return

        self.optimizer.zero_grad()

        current_states = self.batch_samples_muilt_input_tensors([transition[0] for transition in batch])
        new_states = self.batch_samples_muilt_input_tensors([transition[3] for transition in batch])

        current_q_array = self.main_model(*current_states)

        with torch.no_grad():
            future_q_array = self.target_model(*new_states)

        max_future_q = torch.max(future_q_array, dim=1).values
        reward_array = torch.stack([transition[2] for transition in batch])
        new_q = reward_array + self.gamma * max_future_q

        current_qs = current_q_array.clone().detach().requires_grad_(False)
        action_array = torch.stack([transition[1] for transition in batch])
        action_indices = torch.argmax(action_array, dim=1)
        current_qs[range(len(current_qs)), action_indices] = new_q

        loss = self.loss_fn(current_q_array, current_qs)

        loss.backward()
        self.optimizer.step()

        if not self.counter % 5:
            self.target_model.load_state_dict(self.main_model.state_dict())

    def run(self):
        print('Agent is running')

        ''' gamma is initialised here, otherwise it will be lost when process starts'''
        self.gamma = torch.tensor(agent_config.get_gamma()).to(self.device)

        try:
            self.main_model.load_state_dict(torch.load(self.weight_file))
            self.target_model.load_state_dict(torch.load(self.weight_file))
        except (_pickle.UnpicklingError, RuntimeError):
            print('Model params have changed')
        except FileNotFoundError:
            print('Model weight file not found')

        current_state = self.get_data()['state']
        current_state = self.state_np_to_device(current_state)

        while True:
            if np.random.random() > self.epsilon:
                on_device_state = self.state_tensor_to_device(current_state)
                with torch.no_grad():
                    action = self.main_model(*(x.unsqueeze(0) for x in on_device_state)).detach()[0]

                numpy_action = action.cpu().numpy()
            else:
                action_ind = np.random.randint(0, self.key_output_size)
                action = np.zeros(self.key_output_size, dtype=np.float32)
                action[action_ind] = 1

                numpy_action = np.array(action)
                action = torch.tensor(action, dtype=torch.int8).to(self.device)

            predicted_action = self.key_mapping.get_key_from_on_hot_mapping(numpy_action)

            print('\nOutput: {}'.format(numpy_action))
            print('Predicted: {}'.format(predicted_action))

            new_state, reward = self.action(numpy_action)
            new_state = new_state

            new_state = self.state_np_to_device(new_state)
            reward = torch.tensor(reward, dtype=torch.int8).to(self.device)

            self.update_epsilon()

            self.replay_memory.add(current_state, action, reward, new_state)

            current_state = new_state

            self.train()

            self.counter += 1

            if not self.counter % 100:
                self.counter = 0
                torch.save(self.target_model.state_dict(), self.weight_file)
