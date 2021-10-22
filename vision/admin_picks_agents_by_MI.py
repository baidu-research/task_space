import argparse
import numpy as np
import pylab
from numpy.linalg import inv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_mtx', help='path to a .npy of similarity matrix')
    parser.add_argument('--hub-ini', action='store_true',
                        help='use hub as the initial pick')
    parser.add_argument('-k', type=int, default=10,
                        help='number of agents to pick, defaults to 10')
    return parser.parse_args()


def MI_greedy(sim, k, hub_ini):
    """
        Greedy method for picking a subset that maximize its mutual information
        with its complement set
            max_A MI(A; V\A)

        sim: m x m similarity matrix, whose diagonal is all-one
        k: number of items to pick
    """
    m = sim.shape[0]
    picked = []
    increase = []

    for kk in range(k):
        Abar = np.ones(m, dtype=bool)
        Abar[picked] = False
        sigma_A_A = sim[picked][:, picked]
        if kk > 0:
            inv_sigma_A_A = inv(sigma_A_A)
        elif hub_ini: # just use hub as the first one to pick
            include = np.sum(sim, axis=1).argmax()
            Abar_test = np.ones(m, dtype=bool)
            Abar_test[include] = False
            sigma_y_AbarTest = sim[include][Abar_test]
            sigma_AbarTest_AbarTest = sim[Abar_test][:, Abar_test]
            denominator = 1 - sigma_y_AbarTest.dot(inv(sigma_AbarTest_AbarTest)).dot(
                            np.reshape(sigma_y_AbarTest, (-1, 1)))
            delta = 1.0 / denominator
            picked.append(include)
            increase.append(np.log(delta))
            continue

        # loop over unselected agents to find the one that gives biggest gain
        biggest = -1
        for y in range(m):
            if y in picked:
                continue

            if kk == 0:
                numerator = 1
                
            else:
                sigma_y_A = sim[y, picked]
                numerator = 1 - sigma_y_A.dot(inv_sigma_A_A).dot(
                                np.reshape(sigma_y_A, (-1, 1)))

            Abar_test = np.copy(Abar)
            Abar_test[y] = False
            sigma_y_AbarTest = sim[y][Abar_test]
            sigma_AbarTest_AbarTest = sim[Abar_test][:, Abar_test]
            denominator = 1 - sigma_y_AbarTest.dot(inv(sigma_AbarTest_AbarTest)).dot(
                            np.reshape(sigma_y_AbarTest, (-1, 1)))

            delta = numerator / denominator
            if delta > biggest:
                biggest = delta
                include = y
                increase_kk = np.log(delta)
        picked.append(include)
        increase.append(increase_kk)
    return picked, np.cumsum(increase)


if __name__ == '__main__':
    args = parse_args()
    sim = np.load(args.sim_mtx)
    picked, MI = MI_greedy(sim, args.k, args.hub_ini)
    print(' '.join([str(_) for _ in picked]), flush=True)
    #print(MI, flush=True)
    pylab.plot(range(1, args.k+1), MI, 'o-')
    pylab.xlim([1, args.k])
    for i in range(args.k):
        pylab.annotate(str(picked[i]), (1+i, MI[i]))
    pylab.xlabel('# agents')
    pylab.ylabel('MI(selected; not selected)')
    pylab.show()
