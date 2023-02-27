import pandas as pd
from disropt.agents import Agent
from disropt.algorithms import Consensus
from pandas import DataFrame


class LocalVoting(Consensus):

    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        x_avg = self.x + self.gamma * sum([(self.x_neigh[i] - self.x) for i in self.agent.in_neighbors])

        self._update_local_solution(x_avg, **kwargs)


class AgentLB(Agent):
    def __init__(self, produc, queue: DataFrame, **kwargs):
        super(Agent, self).__init__(**kwargs)
        self.produc = produc
        self.queue = queue.sort_values("time").reset_index(drop=True)

    def neighbor_send(self, obj, out_neighbors):
        self.communicator.neighbors_send(obj, out_neighbors)

    def update_value(self, x, step):
        self.execute_tasks()  # todo: move to algorithm and perform before iteration
        self.rearrange_tasks(x, step)
        self.x = self.get_queue_length(step)

    def get_queue_length(self, step):
        return sum(self.get_queue(step).complexity)

    def get_queue(self, step):
        return self.queue[self.queue.time <= step]

    def rearrange_tasks(self, x, step):
        """
        Change number of tasks by x
        :param x: increase by
        :param step: number of current step
        :return:
        """
        if x < 0:
            self.receive_tasks(-x)
        else:
            self.send_tasks(x, step)

    def send_tasks(self, x, step):
        # neibors who vote to receive
        neib_info = self.neighbors_receive_asynchronous()

        if len(neib_info) == 0:
            return

        queue = self.get_queue(step)

        # send tasks
        for key, complex in neib_info.items():
            # extract tasks to send
            num_tasks = 0
            while complex > 0 and queue.shape[0] > 0:
                complex -= queue.iloc[num_tasks].complexity
                num_tasks += 1

            x -= sum(queue.iloc[:num_tasks].complexity)
            send = queue.iloc[:num_tasks]
            self.queue = self.queue[num_tasks:]

            # send tasks
            self.neighbor_send(send, [key])

            if queue.shape[0] == 0 or x < 0:
                break


    def receive_tasks(self, x):
        """
        Say neibors that can get tasks and receive tasks from them
        :param x: number of tasks to receive
        :return:
        """
        self.neighbors_send(x)
        res = self.neighbors_receive_asynchronous()
        for val in res.values():
            self.queue = pd.concat([self.queue.iloc[:1], val, self.queue.iloc[1:]])

    def execute_tasks(self):
        """
        Execute tasks: remove first tasks in the queue with respect to productivity
        :return:
        """
        execute = self.produc
        while execute != 0:
            if self.queue.shape[0] == 0:
                break

            first_task = self.queue.iloc[0, 1]
            if first_task > execute:
                self.queue.iloc[0, 1] = first_task - execute
            else:
                execute -= first_task
                self.queue = self.queue.iloc[1:]


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


class AcceleratedLocalVoting(Consensus):
    def __init__(self, parameters: dict, **kwargs):
        super().__init__(**kwargs)
        self.nesterov_step = self.x

        self.L = parameters.get("L")
        self.mu = parameters.get("mu")
        self.h = parameters.get("h")
        self.eta = parameters.get("eta")
        self.gamma = parameters.get("gamma", [])
        self.alpha = parameters.get("alpha")

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))
        x_n = 1 / (self.gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * self.gamma[0] * self.nesterov_step + self.gamma[1] * self.x)

        step = 0
        for i in self.agent.in_neighbors:
            step += self.agent.in_weights[i] * (x_n - self.x_neigh[i])

        step *= self.h

        self._update_local_solution(x_n - step, **kwargs)
        self.nesterov_step = 1 / self.gamma[0] * \
                             ((1 - self.alpha) * self.gamma[0] * self.nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * step)
