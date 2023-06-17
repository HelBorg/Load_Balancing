import logging
import sys

import numpy as np
import datetime

sys.path.append("C:/Program Files/JetBrains/PyCharm 2021.2.2/debug-eggs/pydevd-pycharm.egg")
from mpi4py import MPI
from lvp.local_voting import AcceleratedLocalVoting, AgentLB
import pandas as pd

parameters = np.load("cache/params.npy", allow_pickle=True)[()]


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
generate = True

# For debugging (start remote debugging 5 times, specify ports, start programm)
# port_mapping = [62383, 54588, 62383, 54590]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
#
# print(f"pid: {os.getpid()}, size = {size}, rank: {rank}")

if __name__ == "__main__":
    # get MPI info
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()
    id = int(datetime.datetime.now().timestamp())
    logging.basicConfig(filename=f'cache/loggs/_loggs_{local_rank}_alvp.log', filemode='w', level=logging.INFO)

    if local_rank == 0:
        print(f"Time: {datetime.datetime.now()}")
        logging.warning(f"Parameters: {parameters}")

    # Generate a common graph
    Adj = np.array([
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0]
    ])
    W = Adj / 2

    # reset local seed
    np.random.seed()

    # generate queue for each agent or use from file
    if generate:
        size = np.random.poisson(lam=50, size=1)[0] + 1
        if local_rank == 5:
            size = 100
        queue_raw = [[np.random.randint(100), np.random.poisson(10)] for i in range(size)]
        add = [[0, np.random.poisson(lam=local_rank*10 + 1, size=1)[0]] for i in range(size)]
        queue_raw = np.append(queue_raw, add, axis=0)
        queue = pd.DataFrame(queue_raw, columns=["time", "complexity"])
        queue.to_csv(f"cache/agent_{local_rank}_queue.csv", index=False)
        logging.warning("Saved initial data")
    else:
        queue = pd.read_csv(f"cache/agent_{local_rank}_queue.csv")

    # create local agent
    agent = AgentLB(queue=queue,
                    produc=5,
                    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
                    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
                    in_weights=W[local_rank, :].tolist())

    # instantiate the consensus algorithm
    d = 1  # decision variable dimension (n, 1)
    algorithm = AcceleratedLocalVoting(
        parameters=parameters,
        agent=agent,
        initial_condition=np.array([0]),
        noise_function=lambda x: 0,
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=100, verbose=True)

    # print solution
    print("Agent {}: {}".format(agent.id, algorithm.get_result()))

    # save data
    np.save("cache/agents.npy", nproc)
    np.save(f"cache/agent_{agent.id}_sequence_alvp.npy", sequence)
    logging.warning(sequence)
