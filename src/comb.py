import os
import time
import random
from numpy.linalg import norm
import numpy as np
from itertools import combinations

data_dir = os.path.join(".\\..\\", "data")
problems = {}
for folder in os.listdir(data_dir):
    problems[folder] = os.path.join(data_dir, folder, folder)

print(problems)
PROBLEM = "berlin52.tsp"

def read(input_name):
    '''
        This function detects whether
    '''
    lines = []
    with open(input_name, "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].strip() == "NODE_COORD_SECTION":
                start = i + 1
            if lines[i].strip() == "EOF":
                end = i
    cords = []
    for i in range(start, end):
        if lines[i][0] == ' ':
            lines[i] = lines[i][1:]
        node_id, x, y = lines[i].split(" ")[:3]
        cords.append([int(node_id), float(x), float(y)])
    return cords

problem = read(problems[PROBLEM])

def dist(node1, node2):
    return norm(np.array(node1[1:]) - np.array(node2[1:]))


def greedy(self):
    cur_node = random.randint(0, self.N - 1)
    solution = [cur_node]

    remain_nodes = set(self.nodes)
    remain_nodes.remove(cur_node)

    while remain_nodes:
        next_node = min(remain_nodes, key=lambda x: dist(self.distance[cur_node], self.distance[x]))
        remain_nodes.remove(next_node)
        solution.append(next_node)
        cur_node = next_node

    cur_total_dis = get_total_dist(self, solution)
    if cur_total_dis < self.best_cost:
        self.best_cost = cur_total_dis
        self.best_tour = solution
        self.cost_history.append(cur_total_dis)
    return solution, cur_total_dis


def get_total_dist(self, tour):
    cur_total_dis = 0
    for i in range(self.N):
        cur_total_dis += dist(self.distance[tour[i % self.N]], self.distance[tour[(i + 1) % self.N]])
    return cur_total_dis

class Tabu:
    def __init__(self, distance, seed=0, limited_time=600):
        # distance input: node_id, x, y
        self.distance = distance
        self.N = len(self.distance)
        self.best_tour = None
        self.best_cost = float("inf")
        self.seed = seed
        self.limited_time = limited_time
        self.nodes = [i for i in range(self.N)]
        self.cost_history = []

    def set_seed(self, seed):
        self.seed = seed

    def tabu(self, curr_tour=None, stopping_criteria=30):
        '''
        Keep tabu list: keep track of recent searches and include
        them into tabu list in order for the algorithm to 'explore'
        different possibilities.
        Steps:
            1. choose a random initial state
            2. enters in a loop checking if a condition to break given
            by the user is met(lower bound)
            3. creates an empty candidate list. Each of the candidates
            in a given neighbor which does not contain a tabu element
            are added to this empty candidate list
            4. It finds the best candidate on this list and if it's cost
            is better than the current best it's marked as a solution.
            5. If the number of tabus on the tabu list have reached the
            maximum number of tabus ( you are defining the number ) a tabu
            expires. The tabus on the list expires in the order they have
            been entered .. first in first out.
        '''
        N = len(self.distance)
        tabu_list = []
        tabu_list_limit = N * 50

        # initialization
        sol_cost = get_total_dist(self, curr_tour)
        neighbor_swap = list(combinations(list(range(N)), 2))

        stop_criterion = 0
        changed = 0
        self.start_time = time.time()
        while time.time() - self.start_time < self.limited_time / 10:
            best_tour, best_cost = [], float("inf")
            # get best solution in the neighbor
            random.shuffle(neighbor_swap)
            for neighbor in neighbor_swap[: len(neighbor_swap) // 3]:
                i, j = neighbor
                # define a neighbor tour
                new_tour = curr_tour.copy()
                new_tour[i: (i + j)] = reversed(new_tour[i: (i + j)])

                new_cost = get_total_dist(self, new_tour)
                if new_cost <= best_cost and new_tour not in tabu_list:
                    best_tour = new_tour
                    best_cost = new_cost

            # stopping criterion:
            if stop_criterion > stopping_criteria and changed <= 10:
                changed += 1
                curr_tour, _ = greedy(self)

            if stop_criterion > stopping_criteria and changed > 10:
                break

            if len(tabu_list) == tabu_list_limit:
                tabu_list.pop()

            if not best_tour:
                best_tour = new_tour  # accpet some worse solution to escape the local maximum
                stop_criterion += 1

            tabu_list.append(best_tour)

            if best_cost < sol_cost:
                curr_tour = best_tour.copy()
                sol_cost = best_cost

        if self.best_cost > sol_cost:
            self.best_cost = sol_cost
            self.best_tour = curr_tour
            self.cost_history.append((round(time.time() - self.start_time, 2), self.best_cost))

    def batch_tabu(self, times=10, stopping_criteria=50):
        start_time = time.time()
        for i in range(1, times + 1):
            random.seed(tabu.seed + i)
            if time.time() - start_time < self.limited_time:
                print(f"Iteration {i}/{times} -------------------------------")
                greedy_tour, _ = greedy(self)
                self.tabu(curr_tour=greedy_tour, stopping_criteria=stopping_criteria)
                print("Best cost obtained: ", self.best_cost)
                print("Best tour", self.best_tour)


tabu = Tabu(problem)


tabu.set_seed(17)
tabu.batch_tabu(times = 10)
print("Best solution", tabu.best_cost)