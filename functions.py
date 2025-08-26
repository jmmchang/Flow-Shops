import numpy as np
from scipy.optimize import linprog
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt

def generate_random_instance(num_jobs = 50, num_machines = 10):
    jobs_data = np.zeros([num_jobs, num_machines], dtype = int)
    for i in range(num_jobs):
        for j in range(num_machines):
            jobs_data[i, j] = np.random.randint(5, 30)

    return jobs_data

def plot(schedule):
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_height = 0.8

    for job_id, ops in schedule.items():
        for machine_id, (st, proc) in ops.items():
            ax.broken_barh(
                [(st, proc)],
                (machine_id - bar_height / 2, bar_height),
                facecolors=f"C{job_id}",
                edgecolor="black",
                label=f"Job{job_id}" if machine_id == 0 else ""
            )

    ax.set_yticks([m for m in range(len(next(iter(schedule.values()))))])
    ax.set_yticklabels([f"M{m}" for m in range(len(next(iter(schedule.values()))))])
    ax.set_xlabel("Time")
    ax.set_title("Flowshop 排程甘特圖")
    ax.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class ConstraintProgramming:
    def __init__(self, jobs_data):
        self.jobs_data = jobs_data.astype(int)

    def run(self):
        n = len(self.jobs_data)
        m = len(self.jobs_data[0])
        horizon = np.sum(self.jobs_data)
        model = cp_model.CpModel()

        start = {}
        end = {}
        interval = {}
        for i in range(n):
            for j in range(m):
                start[i, j] = model.new_int_var(0, horizon, f'start_{i}_{j}')
                end[i, j] = model.new_int_var(0, horizon, f'end_{i}_{j}')
                interval[i, j] = model.new_interval_var(start[i, j], self.jobs_data[i, j], end[i, j], f'interval_{i}_{j}')

        for i in range(n):
            for j in range(m - 1):
                model.add(end[i, j] <= start[i, j + 1])

        for j in range(m):
            model.add_no_overlap([interval[i, j] for i in range(n)])

        makespan = model.new_int_var(0, horizon, 'makespan')
        model.add_max_equality(makespan, [end[i, m - 1] for i in range(n)])
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        result = solver.Solve(model)

        if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f'Makespan: {solver.Value(makespan)}')

            for j in range(m):
                seq = sorted(
                    [(solver.Value(start[i, j]), i) for i in range(n)], key = lambda x: x[0])
                order = [f'Job{i}' for _, i in seq]
                times = [(solver.Value(start[i, j]), solver.Value(end[i, j]))
                         for _, i in seq]
                print(f'\nMachine {j}:')
                for idx, job in enumerate(order):
                    s, e = times[idx]
                    print(f'  {job}: start={s}, end={e}')
        else:
            print('No solution found.')

class Sevastjanov:
    def __init__(self, jobs_data):
        self.jobs_data = jobs_data

    @staticmethod
    def lengthening(jobs_data):
        col_sums = np.sum(jobs_data, axis = 0)
        max_row_sum = np.max(col_sums)
        max_duration = np.max(jobs_data)

        for col in range(jobs_data.shape[1]):
            row = 0
            while col_sums[col] < max_row_sum:
                if jobs_data[row, col] < max_duration:
                    jobs_data[row, col] += 1
                    col_sums[col] += 1
                else:
                    row += 1

    @staticmethod
    def find_permutations_of_vectors(vectors):
        dim = len(vectors[0])
        n = len(vectors)
        remaining_index = list(range(n))
        index_kicked_out = []
        A = np.vstack([np.transpose(np.vstack(vectors)) , np.full(n, 1)])

        for j in range(n, dim, -1):
            b = np.zeros(dim + 1)
            b[-1] = j - dim - 1
            bounds = (0, 1)
            res = linprog(c = np.zeros(j), A_eq = A, b_eq = b, bounds = bounds, method = 'highs')
            lambdas = res.x

            for i in range(len(lambdas)):
                if lambdas[i] == 0:
                    A = np.delete(A, i, axis = 1)
                    index_kicked_out.append(remaining_index[i])
                    remaining_index.pop(i)
                    break

        return remaining_index + index_kicked_out

    @staticmethod
    def flow_shops_makespan(jobs_data):
        n, m = jobs_data.shape[0], jobs_data.shape[1]
        complete_time = [[0]*(m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            for j in range(1, m+1):
                complete_time[i][j] = max(complete_time[i-1][j], complete_time[i][j-1]) + jobs_data[i-1][j-1]

        return complete_time[n][m]

    def run(self):
        n, m = self.jobs_data.shape[0], self.jobs_data.shape[1]
        vectors = np.zeros([n, m - 1])
        self.lengthening(self.jobs_data)

        for i in range(n):
            for j in range(m - 1):
                vectors[i, j] = self.jobs_data[i, j] - self.jobs_data[i, j + 1]

        permutations = self.find_permutations_of_vectors(vectors)

        return self.flow_shops_makespan(self.jobs_data[permutations, :])