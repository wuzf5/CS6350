import numpy as np


if __name__ == "__main__":
    svs = []
    gamma_list = [0.1, 0.5, 1.0, 5.0, 100.0]
    for gamma in gamma_list:
        svs.append(np.load('svs_C{}_gamma{}.npy'.format(0.5727, gamma))) # (n_svs, d)
    def count_overlap(sv1, sv2):
        count = 0
        for svi in sv1:
            for svj in sv2:
                if np.count_nonzero(svi - svj) == 0:
                    count += 1
        return count
    overlap_matrix = []
    for i, sv1 in enumerate(svs):
        overlap_nums = []
        for j, sv2 in enumerate(svs):
            overlap_nums.append(count_overlap(sv1, sv2))
        overlap_matrix.append(overlap_nums)
    overlap_matrix = np.array(overlap_matrix)
    consecutive_overlap = [overlap_matrix[i, i+1] for i in range(len(gamma_list) - 1)]
    print('The number of overlapped support vectors between consecutive gammas:', consecutive_overlap)