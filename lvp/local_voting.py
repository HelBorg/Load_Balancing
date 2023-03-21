import logging

import numpy as np
import pandas as pd
from disropt.agents import Agent
from disropt.algorithms import Consensus
from pandas import DataFrame


class AgentLB(Agent):
    def __init__(self, produc, queue: DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.produc = produc
        self.queue = queue.sort_values("time").reset_index(drop=True)

    def neighbor_send(self, obj, out_neighbors):
        """
        Send data to concrete neighbors
        :param obj:
        :param out_neighbors:
        :return:
        """
        self.communicator.neighbors_send(obj, out_neighbors)

    def update_value(self, old_value, new_value, step):
        self.rearrange_tasks(np.floor(old_value - new_value), step)
        self.execute_tasks(step)

    def get_queue_length(self, step):
        """
        Get queue complexity
        :param step:
        :return:
        """
        return sum(self.get_queue(step).complexity)

    def get_queue(self, step):
        """
        Get queue aat current step
        :return:
        """
        # logging.warning(self.queue[self.queue.time <= step])
        return self.queue[self.queue.time <= step]

    def rearrange_tasks(self, x, step):
        """
        Change number of tasks by x by sending or recieving
        :param x: increase by
        :param step: number of current step
        :return:
        """
        # logging.warning(f"x = {x}, step = {step}")
        if x == 0:
            self.neighbors_exchange(0)
            self.neighbors_exchange(0)
            return
        elif x < 0:
            self.receive_tasks(-x)
        else:
            self.send_tasks(x, step)

    def send_tasks(self, x, step):
        """
        Send x tasks
        :param x:
        :param step:
        :return:
        """
        # get neibors who vote to receive
        neib_info = self.neighbors_exchange(0)
        # logging.warning(f"Will send to {neib_info}")
        if len(neib_info) == 0:
            return

        queue = self.get_queue(step).sort_values("complexity")

        # send tasks
        to_send = {}
        for key, complex in neib_info.items():
            if complex == 0 or x < 0 or queue.shape[0] == 0:
                to_send[key] = pd.DataFrame()
                continue

            # extract tasks to send
            num_tasks = 0
            if abs(complex) > abs(x):
                complex = x

            while complex > 0 and queue.shape[0] > num_tasks:
                complex -= queue.iloc[num_tasks].complexity
                num_tasks += 1

            x -= sum(queue.iloc[:num_tasks].complexity)
            send = queue.iloc[:num_tasks]
            self.queue = self.queue[~self.queue.index.isin(send.index)]
            # logging.warning(f"Send {send} to {key} (needed {complex}), need {x}")
            # send tasks
            to_send[key] = send
        # logging.warning(f"Left \n{self.queue}")

        if len(to_send):
            self.neighbors_exchange(to_send, dict_neigh=True)
            self.queue = self.queue.reset_index(drop=True).sort_values("time")

    def receive_tasks(self, x):
        """
        Tell neibors that can get x tasks and receive tasks from them
        :param x: number of tasks to receive
        :return:
        """
        self.neighbors_exchange(x)
        res = self.neighbors_exchange(0)
        # logging.warning(f"Received res = \n{res}")
        for key, val in res.items():
            if not isinstance(val, pd.DataFrame):
                # logging.error(f"Received not dataframe for {key}: {res}")
                continue

            self.queue = pd.concat([self.queue.iloc[:1], val, self.queue.iloc[1:]])
            # logging.warning(f"Need {x}, get {val} from {key} queu = \n{val}")
        self.queue = self.queue.reset_index(drop=True).sort_values("time")

    def execute_tasks(self, step):
        """
        Execute tasks: remove first tasks in the queue with respect to productivity
        :return:
        """
        execute = self.produc
        # logging.warning(f"Execute {execute}, queue = \n{self.queue}")
        while execute != 0:
            if self.get_queue_length(step) == 0:
                # logging.warning(f"queue is empty could do {execute}")
                break

            first_task = self.queue.iloc[0, 1]
            if first_task > execute:
                self.queue.iloc[0, 1] = first_task - execute
                break
            else:
                execute -= first_task
                self.queue = self.queue.iloc[1:]
        # logging.warning(f"Executed queue {self.queue}")


class LocalVoting(Consensus):

    def __init__(self, gamma, agent: AgentLB, initial_condition: np.ndarray, enable_log: bool = False):
        super(LocalVoting, self).__init__(agent=agent,
                                          initial_condition=initial_condition,
                                          enable_log=enable_log)
        self.gamma = gamma

    def iterate_run(self, step, **kwargs):
        """Run a single iterate of the algorithm
        :param step: current step
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        x_avg = self.x - self.gamma * sum([(self.x - self.x_neigh[i]) for i in self.agent.in_neighbors])
        # logging.warning(f"Step: {step} x: {self.x}, x_avg: {x_avg}")
        self.agent.update_value(self.x, x_avg, step)

    def run(self, iterations: int = 100, verbose: bool = False, **kwargs):
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """
        # logging.warning(f"Agent:{self.agent.id}")
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.x = self.agent.get_queue_length(k)
            # logging.warning(f"Step: {k}, x = {self.x}")
            if k == 0:
                print(f"Agent {self.agent.id}: x = {self.x}")

            if verbose:
                queue_1 = self.agent.get_queue(k)
                new_queue = queue_1[queue_1.time == k]
                # logging.warning(f"new tasks {sum(new_queue.complexity)}: \n{new_queue}")

            self.iterate_run(k, **kwargs)

            if self.enable_log:
                self.x = self.agent.get_queue_length(k)
                self.sequence[k] = self.x


        if self.enable_log:
            return self.sequence


class AccelerateParameters:
    mu = float
    L: float
    gamma: []
    h: float
    eta: float
    a: float

    def from_dict(self, d):
        self.__dict__.update(d)
        return self


class AcceleratedLocalVoting(LocalVoting):
    def __init__(self,
                 parameters: dict,
                 agent: AgentLB,
                 initial_condition: np.ndarray,
                 enable_log: bool = False):
        super(LocalVoting, self).__init__(agent=agent,
                                          initial_condition=initial_condition,
                                          enable_log=enable_log)
        self.nesterov_step = 0

        self.L = parameters.get("L")
        self.mu = parameters.get("mu")
        self.h = parameters.get("h")
        self.eta = parameters.get("eta")
        self.gamma = parameters.get("gamma", [])
        self.alpha = parameters.get("alpha")


    def iterate_run(self, step, **kwargs):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))
        x_n = 1 / (self.gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * self.gamma[0] * self.nesterov_step + self.gamma[1] * self.x)

        y_n = sum([self.agent.in_weights[i] * (x_n - self.x_neigh[i]) for i in self.agent.in_neighbors])
        # logging.warning(f"y_n = {y_n}, x - alpha y_n = {self.x - self.h*y_n}")
        # logging.warning(f"x_n = {x_n} x = {self.x}")
        x_avg = x_n - self.h * y_n

        self.nesterov_step = 1 / self.gamma[0] * \
                             ((1 - self.alpha) * self.gamma[0] * self.nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * y_n)
        # logging.warning(f"Step: {step} x: {self.x}, x_avg: {x_avg}")
        self.agent.update_value(self.x, x_avg, step)

        H = self.h - self.h * self.h * self.L / 2
        if H - self.alpha * self.alpha / (2 * self.gamma[1]) < 0:
            logging.warning(H)
            logging.exception(f"Oh no: {H - self.alpha * self.alpha / (2 * self.gamma[1])}")
            print("Exception")
            raise BaseException()

