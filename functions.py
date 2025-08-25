import numpy as np
from scipy.optimize import linprog

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

def generate_random_instance(num_jobs, num_machines):
    jobs_data = np.zeros([num_jobs, num_machines])

    for i in range(num_jobs):
        for j in range(num_machines):
            jobs_data[i, j] = np.random.randint(5, 30)

    lengthening(jobs_data)
    return jobs_data

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

def flow_shops_makespan(jobs_data):
    n, m = jobs_data.shape[0], jobs_data.shape[1]
    complete_time = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            complete_time[i][j] = max(complete_time[i-1][j], complete_time[i][j-1]) + jobs_data[i-1][j-1]

    return complete_time[n][m]

def run(num_jobs = 50, num_machines = 10):
    jobs_data = generate_random_instance(num_jobs, num_machines)
    n, m = jobs_data.shape[0], jobs_data.shape[1]
    vectors = np.zeros([n, m - 1])

    for i in range(n):
        for j in range(m - 1):
            vectors[i, j] = jobs_data[i, j] - jobs_data[i, j + 1]

    permutations = find_permutations_of_vectors(vectors)

    return round(flow_shops_makespan(jobs_data[permutations, :]))