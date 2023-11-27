import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve
from scipy import fftpack


# ------ 1D SMOOTHING ------
def smooth_1D(array : np.ndarray, smoothing_factor : float = 1, upscale_factor : int = 4):
    lam = smoothing_factor
    k = upscale_factor
    y = array
    m = len(y) - 1

    res_sample = np.linspace(0, m, num = k*(m + 1), endpoint=True)
    res = np.zeros_like(res_sample)

    Rd1 = 2/3 * np.ones(m-1)
    Rd2 = 1/6 * np.ones(m-1)
    R = dia_matrix(([Rd2, Rd1, Rd2], [-1 + i for i in range(3)]), shape=((m-1), (m-1))).tocsr()

    Qd1 = -2 * np.ones(m-1)
    Qd2 = np.ones(m-1)
    Q = dia_matrix(([Qd2, Qd1, Qd2], [-2 + i for i in range(3)]), shape=((m+1), (m-1))).tocsc()
    Q_T = Q.transpose()

    Ad1 = (2/3 + 6 * lam) * np.ones((m-1))
    Ad2 = (1/6 - 4 * lam) * np.ones((m-1))
    Ad3 = lam * Qd2

    A = dia_matrix(([Ad3, Ad2, Ad1, Ad2, Ad3], [-2 + i for i in range(5)]), shape=((m-1), (m-1))).tocsc()

    sig = spsolve(A, Q_T @ y)
    sig = np.insert(np.zeros(2), 1, sig)

    s = y - lam * Q @ sig[1:-1]

    eval = lambda x, i : (x - (i-1)) * s[i] + (i - x) * s[i-1] - 1/6 * (x - (i-1)) * (i - x) * ((2 + x - i) * sig[i] + (1 + i - x) * sig[i-1])

    for index in range(len(res_sample) - 1):
        x = res_sample[index]
        i = int(np.floor(x)) + 1
        res[index] = eval(x, i)
    
    res[-1] = eval(res_sample[-1], i)

    return res


lambdas = np.array([0.01 * (2**i) for i in range(18)])

upscale_factor = 4

I = plt.imread('Homework3/data/boat.512.tiff')[120:220, 240:340]
plt.imsave('Homework3/images/boat_orig.tiff', I, cmap='grey')

I_inter = np.zeros((len(I), upscale_factor * len(I[0])))
I_new = np.zeros((upscale_factor * len(I), upscale_factor * len(I[0])))

res = np.zeros_like(lambdas)

for index, lam in enumerate(lambdas):
    print(index)
    for i in range(len(I)):
        I_inter[i] = smooth_1D(I[i], upscale_factor=upscale_factor, smoothing_factor=lam)

    for i in range(len(I_inter[0])):
        I_new[:, i] = smooth_1D(I_inter[:, i], upscale_factor=upscale_factor, smoothing_factor=lam)

    fft2 = fftpack.fft2(I_new)

    res[index] = np.sum(np.abs(fft2))

    plt.imsave(f"Homework3/images/boat_lambda_{lam}.tiff", I_new, cmap='grey')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(lambdas, res, 'k')
ax.grid(which='both', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$l_1$-norm')
ax.set_title('Norm of the Fourier transform of the image')
fig.savefig('Homework3/images/fftnorm_log.eps', format='eps')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(lambdas, res, 'k')
ax.grid(which='both', alpha=0.5)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$l_1$-norm')
ax.set_title('Norm of the Fourier transform of the image')
fig.savefig('Homework3/images/fftnorm.eps', format='eps')


plt.show()
