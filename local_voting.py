import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("C:/Program Files/JetBrains/PyCharm 2021.2.2/debug-eggs/pydevd-pycharm.egg")
import pydevd_pycharm
from disropt.agents import Agent
from mpi4py import MPI
from lvp.local_voting import LocalVoting, AgentLB

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
generate = True
# port_mapping = [57380, 54588, 54589, 54590]
# if rank < 1:
#     pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
#
# print(f"pid: {os.getpid()}, size = {size}, rank: {rank}")

# mpiexec -np 3 python local_voting.py

if __name__ == "__main__":
    # get MPI info
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()
    logging.basicConfig(filename=f'cache/loggs_{local_rank}.log', filemode='w', level=logging.INFO)

    # Generate a common graph (everyone use the same seed)
    Adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    W = Adj / 2

    # reset local seed
    np.random.seed()

    if generate:
        queue_raw = np.random.randint(0, 100, size=(100, 2))
        add = [[0, np.random.randint(100)] for i in range(10)]
        queue_raw = np.append(queue_raw, add, axis=0)

        queue = pd.DataFrame(queue_raw, columns=["time", "complexity"])
        queue.to_csv(f"cache/agent_{local_rank}_queue.csv", index=False)
    else:
        queue = pd.read_csv(f"cache/agent_{local_rank}_queue.csv")

    logging.info(f"Queue: \n{queue}")
    # create local agent
    agent = AgentLB(queue=queue,
                    produc=1,
                    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
                    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
                    in_weights=W[local_rank, :].tolist())
    # instantiate the consensus algorithm
    d = 2  # decision variable dimension (n, 1)

    algorithm = LocalVoting(
        gamma=0.25,
        agent=agent,
        initial_condition=np.array([0]),
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=100)

    # print solution
    print("Agent {}: {}".format(agent.id, algorithm.get_result()))

    # save data
    np.save("cache/agents.npy", nproc)
    np.save("cache/agent_{}_sequence_lvp.npy".format(agent.id), sequence)
    logging.warning(sequence)
