import numpy as np
from scipy.optimize import linprog
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import seaborn as sns

def generate_random_instance(num_jobs = 200, num_machines = 5):
    jobs_data = np.zeros([num_jobs, num_machines], dtype = int)
    for i in range(num_jobs):
        for j in range(num_machines):
            jobs_data[i, j] = np.random.randint(5, 51)

    return jobs_data

class ConstraintProgramming:
    def __init__(self, jobs_data, plot = False):
        self.jobs_data = jobs_data.astype(int)
        self.plot = plot

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
        solver.parameters.max_time_in_seconds = 600
        result = solver.Solve(model)

        if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f'Makespan: {solver.Value(makespan)}')

        if self.plot:
            schedule = {}
            for i in range(n):
                schedule[i] = {}
                for j in range(m):
                    st = solver.Value(start[i, j])
                    dur = self.jobs_data[i, j]
                    schedule[i][j] = (st, dur)

            jobs = list(schedule.keys())
            fig, ax = plt.subplots(figsize=(20, 16))
            palette = sns.color_palette("hls", len(jobs))
            job_colors = dict(zip(jobs, palette))
            bar_height = 0.8
            for job_id, ops in schedule.items():
                for machine_id, (st, dur) in ops.items():
                    ax.broken_barh(
                        [(st, dur)],
                        (machine_id - bar_height / 2, bar_height),
                        facecolors = job_colors[job_id],
                        edgecolor = "black",
                        label = (f"Job{job_id}" if machine_id == 0 else "")
                    )

            ax.set_yticks(range(m))
            ax.set_yticklabels([f"M{j}" for j in range(m)])
            ax.set_xlabel("Time")
            ax.set_ylabel("Machine")
            ax.set_title("Flow-Shops Gantt Chart")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc = "upper left")
            plt.show()

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
    def find_permutation_of_vectors(vectors):
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

        permutations = self.find_permutation_of_vectors(vectors)

        return int(self.flow_shops_makespan(self.jobs_data[permutations, :]))