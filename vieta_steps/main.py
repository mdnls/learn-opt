import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def symm_poly(inp):
    '''
    Evaluate the elementary symmetric polynomials wrt inp

    :param i: integer
    :param inp: a list
    :param cumulative: if true, return a list of length i of all symmetric polynomials up to
    :return: elementwise evaluation of ei
    '''

    outp = np.ones((len(inp) + 1, len(inp) + 1))

    for k in range(1, len(inp) + 1):
        for n in range(1, k):
            outp[k, n] = outp[k - 1, n] + outp[k - 1, n - 1] * inp[k - 1]
        outp[k, k] = inp[k - 1] * outp[k - 1, k - 1]

    return outp[-1]

def mtx_poly(A, x, coeff):
    '''
    If D=len(coeff), evaluate the polynomial (in A) applied to columns of X.

    :param A: N by M matrix
    :param x: [..., M] tensor
    :param coeff: [K,] floats
    :return:
    '''

def regr_to_steps(regr):
    '''
    convert regression coefficients to step sizes
    '''
    return np.roots(regr)


def steps_to_regr(steps):
    '''
    convert step sizes to regression coefficients
    '''
    return (- np.ones(len(steps) + 1))**(np.arange(len(steps) + 1)) * symm_poly(steps)

def mtx_optimal_regr(A, k):
    '''
    compute the optimal regression coefficients for a matrix A
    '''
    eigs = np.linalg.eigvalsh(A)
    powers = np.stack([eigs**j for j in range(k+1)], axis=0)
    mtx = np.linalg.inv(powers @ powers.T)
    return mtx[:,0] / mtx[0, 0]

def obj_value(A, steps):
    return regr_obj_value(A, steps_to_regr(steps)[1:])

def regr_obj_value(A, coeffs):
    eigs = np.linalg.eigvalsh(A)
    powers = np.stack([eigs**j for j in range(len(coeffs)+1)], axis=0)
    return np.sum((powers.T @ np.concatenate(([1], coeffs)))**2)

def gd_trajectory(A, x_0, steps):
    '''
    compute the trajectory of gradient descent initialized at x_0, with step sizes steps
    x_0 is an N by D array of initializations
    '''
    X = [x_0]
    for i in range(len(steps)):
        X.append(X[-1] - steps[i] * X[-1] @ A.T)
    return np.transpose(np.stack(X, axis=0), (1, 0, 2))


def trajectory_evecs_superplot(proj, trajectories):
    '''
    :param proj: an ordered list of projection vectors of size [k eigenvectors x d dimension]
    :param trajectories: a list of size N x steps x d of trajectories
    :return:
    '''
    K = len(proj) - 1
    N = len(trajectories)
    N_steps = trajectories.shape[1]

    cmap = mpl.colormaps.get_cmap('tab20c')

    for k in range(K):
        cur_proj = proj[k:k+2]
        for n in range(1, N_steps-1):
            plt.subplot(K, N_steps, k*N_steps + n + 1)
            # make the plot ...
            for i, traj in enumerate(trajectories):
                if n >= 2:
                    pre_traj = trajectories[i:i+1, :n-1]
                    plt.plot(*(pre_traj @ cur_proj.T).T, marker='o', color=cmap(4*i+3))
                plt.plot(*(trajectories[i:i+1, n-1:n+1] @ cur_proj.T).T, marker='x', color=cmap(4*i))