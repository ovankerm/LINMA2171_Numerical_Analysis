from matplotlib import pyplot as plt
import numpy as np
from time import perf_counter
from numpy.random import normal
from numpy.polynomial.legendre import legvander, Legendre
from numpy.linalg import norm, eigvals

# ---------------------------------------
#         QUESTION 1
# ---------------------------------------

# ------ CLASS AND FUNCTIONS ------
class Neville:
    def __init__(self, x : np.ndarray, y : np.ndarray) -> None:
        self.x = x
        self.y = y

    def __call__(self, x : np.ndarray) -> np.ndarray:
        temp = np.array([self.y for i in range(len(x))]).T
        N = len(self.x)
        for i in range(1, N):
            for j in range(0, N - i):
                start = j
                end = j + i
                temp[j] = ((x - self.x[start]) * temp[j + 1] - (x - self.x[end]) * temp[j])/(self.x[end] - self.x[start])

        return temp[0]


class Newton:
    def __init__(self, x : np.ndarray, y : np.ndarray) -> None:
        self.x = x
        self.y = y
        self.coeff = np.zeros_like(x)
        self.set_coeff()

    def set_coeff(self):
        temp = np.copy(self.y)
        N = len(self.x)
        self.coeff[0] = self.y[0]
        for i in range(1, N):
            for j in range(0, N - i):
                start = j
                end = j + i
                temp[j] = (temp[j+1] - temp[j])/(self.x[end] - self.x[start])
            self.coeff[i] = temp[0]
    
    def __call__(self, x : np.ndarray) -> np.ndarray:
        res = self.coeff[-1]
        for i in range(len(self.coeff) - 1):
            res *= x - self.x[-(i+2)]
            res += self.coeff[-(i+2)]
        return res

class Polyfit:
    def __init__(self, x : np.ndarray, y : np.ndarray) -> None:
        self.p = np.poly1d(np.polyfit(x, y, deg = len(x)))

    def __call__(self, x) -> np.ndarray:
        return self.p(x)
    
def time_algo(x, y, x_eval, solver = None, algo : str = None):
    if algo is None and solver is None : raise(ValueError('solver and algo cannot be None'))

    t_start_init = perf_counter()
    if solver == None:
        if algo =='neville':
            p = Neville(x, y)
        elif algo == 'polyfit':
            p = Polyfit(x, y)
        else:
            p = Newton(x, y)
    else: p = solver

    t_start_eval = perf_counter()

    res = p(x_eval)
    

    t_end = perf_counter()

    return res, (t_start_eval - t_start_init), (t_end - t_start_eval)


# ------ FUNCTION DEFINITION ------
f = lambda x : np.sin(10 * x) + 2 * x

# ------ Plots Newton ------
N_1 = 8
N_2 = 14
x_1 = np.linspace(-1, 1, num = N_1, endpoint=True)
y_1 = f(x_1)
x_2 = np.linspace(-1, 1, num = N_2, endpoint=True)
y_2 = f(x_2)

x_plot = np.linspace(-1, 1, num = 200, endpoint=True)

new_1 = Newton(x_1, y_1)
new_2 = Newton(x_2, y_2)


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.grid()
ax.plot(x_plot, f(x_plot), 'g-', label='target function')
ax.plot(x_plot, new_1(x_plot), 'k-', label=f'n = {N_1 - 1}')
ax.plot(x_plot, new_2(x_plot), 'k--', label=f'n = {N_2 - 1}')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Newton's algorithm")

fig.savefig("Homework2/images/Newton.eps", format='eps')

# ------ Plots Neville ------
N_1 = 6
N_2 = 17
x_1 = np.linspace(-1, 1, num = N_1, endpoint=True)
y_1 = f(x_1)
x_2 = np.linspace(-1, 1, num = N_2, endpoint=True)
y_2 = f(x_2)

x_plot = np.linspace(-1, 1, num = 200, endpoint=True)

nevil_1 = Neville(x_1, y_1)
nevil_2 = Neville(x_2, y_2)

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.grid()
ax.plot(x_plot, f(x_plot), 'g-', label='target function')
ax.plot(x_plot, nevil_1(x_plot), 'k-', label=f'n = {N_1 - 1}')
ax.plot(x_plot, nevil_2(x_plot), 'k--', label=f'n = {N_2 - 1}')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Neville's algorithm")

fig.savefig("Homework2/images/Neville.eps", format='eps')

# ------ FIXED m ------
algos = ['newton', 'neville', 'polyfit'] 
m = 10
N = np.array([50, 100, 200, 400, 800])
x_eval = np.linspace(-1, 1, num=m, endpoint=True)
times_init = np.zeros((len(algos), len(N)))
times_eval = np.zeros((len(algos), len(N)))

for j, algo in enumerate(algos):
    for i, n in enumerate(N):
        print(f"algo : {algo} with N = {n}")
        x = np.linspace(-1, 1, endpoint=True, num=n)
        y = f(x)
        times_init[j, i], times_eval[j, i] = time_algo(x, y, x_eval, algo=algo)[1:]

times_all = times_eval + times_init


linestyles = ['k', 'b', 'g']
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
for i in range(3): 
    ax.plot(N, times_all[i], linestyles[i], label = algos[i])
    print(f" ------ {algos[i]} ------")
    print(f"Order of {np.polyfit(np.log(N[2:]), np.log(times_all[i, 2:]), 1)[0]} for the entire eval")
    print(f"Order of {np.polyfit(np.log(N[2:]), np.log(times_init[i, 2:]), 1)[0]} for the initiation")
    print(f"Order of {np.polyfit(np.log(N[2:]), np.log(times_eval[i, 2:]), 1)[0]} for the evaluation")
    print()
ax.legend()
ax.grid(which='both')
ax.set_yscale('log')
ax.set_xlabel('N')
ax.set_ylabel('time [s]')
ax.set_title("Fixed m")

fig.savefig("Homework2/images/fixed_m.eps", format='eps')



# ------ Fixed n ------
N = 51
x = np.linspace(-1, 1, endpoint=True, num=N)
y = f(x)
solvers = [Newton(x, y), Neville(x, y), Polyfit(x, y)]
ms = [25, 50, 100, 200]
times_init = np.zeros((len(algos), len(ms)))
times_eval = np.zeros((len(algos), len(ms)))
for i, solver in enumerate(solvers):
    for j, m in enumerate(ms):
        print(f"algo : {algos[i]} with m = {m}")
        x_eval = np.linspace(-1, 1, num=m, endpoint=True)
        times_init[i, j], times_eval[i, j] = time_algo(x, y, x_eval, solver=solver)[1:]

times_all = times_eval + times_init

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
for i in range(3): ax.plot(ms, times_all[i], linestyles[i], label = algos[i])
ax.legend()
ax.grid(which='both')
ax.set_yscale('log')
ax.set_xlabel('m')
ax.set_ylabel('time [s]')
ax.set_title(f"Fixed n = {N - 1}")

fig.savefig("Homework2/images/fixed_n.eps", format='eps')


# ------ RESULTS ------
#  ------ newton ------
# Order of 1.9730201433444372 for the entire eval
# Order of 1.993086190830223 for the initiation
# Order of 1.0047804778136216 for the evaluation

#  ------ neville ------
# Order of 1.9707979630338848 for the entire eval
# Order of 0.4627265233858558 for the initiation
# Order of 1.970802619595436 for the evaluation

#  ------ polyfit ------
# Order of 2.6828943478831593 for the entire eval
# Order of 2.706867639355788 for the initiation
# Order of 0.9756830627393066 for the evaluation


# ---------------------------------------
#         QUESTION 2
# ---------------------------------------

# ------ FUNCTIONS AND INIT ------
def get_error(f_exact, f_approx):
    M = 100
    x_test = np.linspace(-1, 1, num=M, endpoint=True)
    n = norm(f_exact(x_test) - f_approx(x_test))
    return n*n/M
    
N = 20
x = np.linspace(-1, 1, num=20, endpoint=True)
f = np.poly1d([1, -2, 5, 5, -6, 1])
y = f(x) + normal(0, 0.2, N)

def compute_all(P):
    phi = legvander(x, P - 1)
    grad = lambda beta : 2 * (phi.T @ phi) @ beta - 2 * phi.T@y
    L = np.max(eigvals(2 * phi.T@phi))
    eta = 1/L
    count = 0
    t_start = perf_counter()
    b = np.zeros(P)
    while(norm(grad(b)) > 1e-4):
        b = b - eta * grad(b)
        count += 1
    t_end = perf_counter()
    p_res = Legendre(b.real)
    err = get_error(f, p_res)

    return err, p_res, t_end - t_start, count

# ------ COMPUTATIONS ------
Ps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 100])
to_plot = [4, 18, 27, 33]
errors = np.zeros_like(Ps, dtype=float)
approx = np.empty(len(Ps), dtype=Legendre)
times = np.zeros(len(Ps))
iters = np.zeros(len(Ps))
for i,P in enumerate(Ps):
    print(P)
    errors[i], approx[i], times[i], iters[i] = compute_all(P)

# ------ PLOTS ------
x_plot = np.linspace(-1, 1, num=200, endpoint=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.grid()
ax.plot(Ps[:-1], errors[:-1], 'k')
ax.set_yscale('log')
ax.set_xlabel('P')
ax.set_ylabel('testing error')
ax.set_title('Testing error in terms of the number of parameters P')
fig.savefig('Homework2/images/testing_error.eps', format='eps')

fig, ax = plt.subplots(2, 2, figsize = (15, 8))

for index, k in enumerate(to_plot):
    i = index//2
    j = index%2
    ax[i,j].grid()
    ax[i,j].scatter(x, y, c='k')
    ax[i,j].plot(x_plot, f(x_plot), 'g', label=r'$f(x)$')
    ax[i,j].plot(x_plot, approx[k](x_plot), 'k', label=r'$p(x)$')
    ax[i,j].legend()
    ax[i,j].set_title(f'P = {Ps[k]}')
    ax[i,j].set_xlabel('x')
    ax[i,j].set_ylabel('y')
    print(f'For P = {Ps[k]}, error = {errors[k]}, iterations = {iters[k]}, time = {times[k]}')

fig.suptitle('Approximants for different values of P')
fig.savefig('Homework2/images/approx.eps', format='eps')

plt.show()
