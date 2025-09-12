import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function
from Py_FS.wrapper.nature_inspired._utilities import (
    Data,
    Solution,
    compute_fitness,
    display,
    initialize,
    sort_agents,
)
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "IBM Plex Mono"
CONV_PLOT = plt.figure(num=1)


def AchimedesOptimizer(
    num_agents,
    max_iter,
    train_data,
    train_label,
    obj_function=compute_fitness,
    trans_function_shape="s",
    save_conv_graph=False,
    val_size=0.3,
    weight_acc=0.9,
    save_dir="",
):
    agent_name = "CBO"
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)

    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1)

    crystals = initialize(num_agents, num_features)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    convergence_curve = {"fitness": np.zeros(max_iter)}

    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(
        train_data, train_label, stratify=train_label, test_size=val_size
    )

    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    crystals, fitness = sort_agents(crystals, obj, data)

    start_time = time.time()

    for iter_no in range(max_iter):
        print(
            "================================================================================"
        )
        print("                          Iteration - {}".format(iter_no + 1))
        print(
            "================================================================================"
        )

        a = 2 - iter_no * (2 / max_iter)
        for i in range(num_agents):
            r = np.random.random()
            A = (2 * a * r) - a
            C = 2 * r
            l_ = -1 + (np.random.random() * 2)
            p = np.random.random()
            b = 1

            if p < 0.5:
                if abs(A) >= 1:
                    rand_agent_index = np.random.randint(0, num_agents)
                    rand_agent = crystals[rand_agent_index, :]
                    mod_dist_rand_agent = abs(C * rand_agent - crystals[i, :])
                    crystals[i, :] = rand_agent - (A * mod_dist_rand_agent)  # Eq. (9)

                else:
                    mod_dist_Leader = abs(C * Leader_agent - crystals[i, :])
                    crystals[i, :] = Leader_agent - (A * mod_dist_Leader)  # Eq. (2)

            else:
                dist_Leader = abs(Leader_agent - crystals[i, :])
                crystals[i, :] = (
                    dist_Leader * np.exp(b * l_) * np.cos(l_ * 2 * np.pi) + Leader_agent
                )

            for j in range(num_features):
                trans_value = trans_function(crystals[i, j])
                if np.random.random() < trans_value:
                    crystals[i, j] = 1
                else:
                    crystals[i, j] = 0

        crystals, fitness = sort_agents(crystals, obj, data)
        display(crystals, fitness, agent_name)
        if fitness[0] > Leader_fitness:
            Leader_agent = crystals[0].copy()
            Leader_fitness = fitness[0].copy()

        convergence_curve["fitness"][iter_no] = np.mean(fitness)

        if save_conv_graph:
            cv = [convergence_curve["fitness"][i] for i in range(iter_no + 1)]
            num_iter = len(cv)
            iters = np.arange(num_iter) + 1
            ax = CONV_PLOT.gca()
            ax.clear()
            ax.set_title("Convergence of Fitness over Iterations", pad=10)
            ax.set_xlabel("Iteration", labelpad=10)
            ax.set_ylabel("Avg. Fitness", labelpad=10)
            ax.plot(iters, cv, color="tab:orange")
            ax.grid(True)
            CONV_PLOT.tight_layout()
            CONV_PLOT.savefig(os.path.join(save_dir, "convergence_graph.jpg"))

    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    crystals, accuracy = sort_agents(crystals, compute_accuracy, data)

    print(
        "\n================================================================================"
    )
    print(
        "                                    Final Result                                  "
    )
    print(
        "================================================================================"
    )
    print("Leader " + agent_name + " Dimension : {}".format(int(np.sum(Leader_agent))))
    print("Leader " + agent_name + " Fitness : {}".format(Leader_fitness))
    print(
        "Leader " + agent_name + " Classification Accuracy : {}".format(Leader_accuracy)
    )
    print(
        "================================================================================"
    )

    end_time = time.time()
    exec_time = end_time - start_time

    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_population = crystals
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution
