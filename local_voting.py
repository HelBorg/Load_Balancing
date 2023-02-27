import os
import sys

import numpy as np

sys.path.append("C:/Program Files/JetBrains/PyCharm 2021.2.2/debug-eggs/pydevd-pycharm.egg")
import pydevd_pycharm
from disropt.agents import Agent
from mpi4py import MPI
from lvp.local_voting import LocalVoting

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
# port_mapping = [57380, 54588, 54589, 54590]
# if rank < 1:
#     pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
#
# print(f"pid: {os.getpid()}, size = {size}, rank: {rank}")

if __name__ == "__main__":
    # get MPI info
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    # Generate a common graph (everyone use the same seed)
    Adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    W = Adj / 2

    # reset local seed
    np.random.seed()

    # create local agent
    agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
                  out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
                  in_weights=W[local_rank, :].tolist())

    # instantiate the consensus algorithm
    d = 2  # decision variable dimension (n, 1)
    x0 = np.array([{
                       0: 0,
                       1: 50,
                       2: 100
                   }[rank]])

    algorithm = LocalVoting(
        gamma=0.25,
        agent=agent,
        initial_condition=x0,
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=20)

    # print solution
    print("Agent {}: {}".format(agent.id, algorithm.get_result()))

    # save data
    np.save("cache/agents.npy", nproc)
    np.save("cache/agent_{}_sequence.npy".format(agent.id), sequence)
