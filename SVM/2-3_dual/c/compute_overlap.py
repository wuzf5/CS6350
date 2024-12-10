import numpy as np


if __name__ == "__main__":
    svs = []
    gamma_list = [0.1, 0.5, 1.0, 5.0, 100.0]
    # Cs = [0.1145, 0.5727, 0.8018]
    li = []
    # for C in Cs:
    for gamma in gamma_list:
        svs.append(np.load('svs_C{}_gamma{}.npy'.format(0.5727, gamma))) # (n_svs, d)
    #         d = np.load('svs_C{}_gamma{}.npy'.format(C, gamma))
    #         li.append(d.shape[0])
    #         li.append('&')
    # print(li)
    def count_overlap(sv1, sv2):
        count = 0
        for i, svi in enumerate(sv1):
            for j, svj in enumerate(sv2):
                if (svi==svj).all():
                    count += 1
                    break
        return count
    overlap_matrix = []
    for i, sv1 in enumerate(svs):
        overlap_nums = []
        # for j, sv2 in enumerate(svs):
        if i < len(svs) - 1:
            overlap_nums.append(count_overlap(sv1, svs[i+1]))
        overlap_matrix.append(overlap_nums)
    # overlap_matrix = np.array(overlap_matrix)
    # consecutive_overlap = [overlap_matrix[i, i+1] for i in range(len(gamma_list) - 1)]
    print('The number of overlapped support vectors between consecutive gammas:', overlap_matrix)