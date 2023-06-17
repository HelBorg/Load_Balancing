import logging
from datetime import datetime

import numpy as np
import pandas as pd
from disropt.agents import Agent
from disropt.functions import QuadraticForm, Variable
from disropt.problems import Problem
from disropt.utils.graph_constructor import binomial_random_graph
from mpi4py import MPI

# get MPI info
from lvp.local_voting import ADMM_LB, AgentLB

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
generate = False

# For debugging
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

    logging.basicConfig(filename=f'cache/loggs/_loggs_{local_rank}.log', filemode='w', level=logging.INFO)

    if local_rank == 0:
        print(f"Time: {datetime.now()}")
    # Generate a common graph (everyone use the same seed)
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

    if generate:
        queue_raw = np.random.randint(0, 100, size=(20, 2))
        add = [[0, np.random.randint(100)] for i in range(10)]
        queue_raw = np.append(queue_raw, add, axis=0)

        queue = pd.DataFrame(queue_raw, columns=["time", "complexity"])
        queue.to_csv(f"cache/agent_{local_rank}_queue.csv", index=False)
    else:
        queue = pd.read_csv(f"cache/agent_{local_rank}_queue.csv")

    # logging.info(f"Queue: \n{queue}")
    # create local agent
    agent = AgentLB(queue=queue,
                    produc=5,
                    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
                    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
                    in_weights=W[local_rank, :].tolist())
    # instantiate the consensus algorithm
    if local_rank == 0:
        logging.info(f"Looking for: {np.nonzero(Adj[local_rank, :])[0].tolist()} {np.nonzero(Adj[:, local_rank])[0].tolist()} {W[local_rank, :].tolist()}")

    # variable dimension
    n = len(np.nonzero(Adj[local_rank, :])[0].tolist())
    print(f"N = {n}")

    # generate a positive definite matrix
    P = np.eye(n)

    # declare a variable
    x = Variable(n)

    # define the local objective function
    fn = QuadraticForm(x, P)

    # define a (common) constraint set
    constr = [np.ones((n, 1)) @ x == 10]

    # local problem
    pb = Problem(fn, constr)
    agent.set_problem(pb)

    # instantiate the algorithms
    initial_z = np.ones((n, 1))
    # initial_lambda = {local_rank: 10*np.random.rand(n, 1)}
    initial_lambda = {local_rank: np.ones((n, 1))}

    for j in agent.in_neighbors:
        # initial_lambda[j] = 10*np.random.rand(n, 1)
        initial_lambda[j] = np.ones((n, 1))

    algorithm = ADMM_LB(agent=agent,
                        initial_lambda=initial_lambda,
                        initial_z=initial_z,
                        enable_log=True)

    # run the algorithm
    x_sequence, lambda_sequence, z_sequence = algorithm.run(iterations=100, penalty=0.1, verbose=True)
    x_t, lambda_t, z_t = algorithm.get_result()

    # print solution
    print("Agent {}: primal {} dual {} auxiliary primal {}".format(agent.id, x_t.flatten(), lambda_t, z_t.flatten()))

    # save data
    np.save("cache/agents.npy", nproc)
    np.save("cache/agent_{}_sequence_admm.npy".format(agent.id), x_sequence)
    logging.warning(x_sequence)



