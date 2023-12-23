import numpy as np
from numpy.linalg import solve
from matplotlib import pyplot as plt
from time import perf_counter

# print(np.pi/4)

# ------ ALGORITHMS ------
def uniform_search_grid(g, N_grid, a, b):
    dx = (b - a)/(N_grid - 1)
    eps = 1e-3 * dx
    tol = 1e-16

    d = {}

    d[a] = abs(g(a))
    d[b] = abs(g(b))

    for i in range(N_grid - 1):
        x_1 = a + i * dx
        x_2 = a + (i+1) * dx
        g_prime_1 = g(min(b, x_1 + eps)) - g(max(a, x_1 - eps))
        g_prime_2 = g(min(b, x_2 + eps)) - g(max(a, x_2 - eps))
        if abs(g_prime_1) < 1e-16:
            d[x_1] = abs(g(x_1))
            break
        if abs(g_prime_2) < 1e-16:
            d[x_2] = abs(g(x_2))
            break
        if(g_prime_1 * g_prime_2 <= 0):
            while(abs(x_1 - x_2) > tol):
                h = x_2-x_1
                x_mid = 0.5 * (x_1 + x_2)
                g_prime_mid = g(min(b, x_mid + 1e-3 * h)) - g(max(a, x_mid - 1e-3 * h))
                if g_prime_mid ==0:
                    break
                if(g_prime_1 * g_prime_mid < 0):
                    x_2 = x_mid
                    g_prime_2 = g_prime_mid
                else:
                    x_1 = x_mid
                    g_prime_1 = g_prime_mid

            d[x_mid] = abs(g(x_mid))

    x_max = max(d, key=d.get)
    return x_max, d[x_max]

def Remez_exchange(f, N_grid, a, b, n, tol, p_ref = 'equi'):
    if p_ref == 'equi':
        xi = np.linspace(a, b, num=n+2, endpoint=True)
    elif p_ref == 'cheby':
        xi = -np.cos(np.pi * np.arange(n+2)/(n+1))

    A = np.zeros((n+2, n+2))
    A[:, -1] = np.power(-1, np.arange(n+2))

    res = 10 * tol
    h = tol

    while(res - abs(h) > tol):
        for i in range(n+1):
            A[:, i] = np.power(xi, i)
        x = solve(A, f(xi))

        h = x[-1]
        coeff = x[:-1]

        p = np.polynomial.polynomial.Polynomial(coeff)
        g = lambda x : f(x) - p(x)

        eta, res = uniform_search_grid(g, N_grid, a, b)

        if eta < xi[0] and g(eta) * g(xi[0]) > 0:
            xi[0] = eta
        elif eta > xi[-1] and g(eta) * g(xi[-1]) > 0:
            xi[-1] = eta
        else:
            for i in range(len(xi) - 1):
                if xi[i] < eta and eta < xi[i+1]:
                    if g(eta) * g(xi[i]) > 0:
                        xi[i] = eta
                    elif g(eta) * g(xi[i+1]) > 0:
                        xi[i+1] = eta
                    else:
                        return p
                    break

    return p



# ------ FUNCTIONS ------
f1 = lambda x : 1/(1 + np.exp(-10*x))
f2 = lambda x : np.sqrt(1 + 1e-4 - x)

#------ FIRST ALGO ------
Ns = [101]
offsets = [0, np.pi/4, np.pi/40, np.pi/400]

print()
for N in Ns:
    for off in offsets:
        f = lambda x : np.exp(-(x-off)**2)
        start = perf_counter()
        (x_max, m) = uniform_search_grid(f, N, -1, 1)
        end = perf_counter()

        print("Max in %.4e with value %.4e for N_grid = %d and a = %.4e, errors in abs : %.4e in val : %.4e; time : %.4e"%(x_max, m, N, off, abs(min(1, off) - x_max), abs(f(min(1, off)) - m), end-start))


print()
f = lambda x : x
start = perf_counter()
(x_max, m) = uniform_search_grid(f, N, -1, 1)
end = perf_counter()
print("Max in %.4e with value %.4e errors in abs : %.4e in val : %.4e"%(x_max, m, abs(1 - abs(x_max)), abs(1 - abs(m))))

print()
f = lambda x : np.sin(10 * np.pi * x)
start = perf_counter()
(x_max, m) = uniform_search_grid(f, N, -1, 1)
end = perf_counter()
print("Max in %.4e with value %.4e errors in abs : %.4e in val : %.4e"%(x_max, m, abs(0.95 - abs(x_max)), abs(1 - abs(m))))


# ------ SECOND ALGO ------
ns = [1, 2, 3, 8]
ls = ['-', '--', '-.', ':']
x_plot = np.linspace(-1, 1, num=2000)


print()
f = f1
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'equi')
    end = perf_counter()
    print(f'Time for n = {n} and equidistant points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with equidistant points')
fig1.savefig('Homework4/images/f1_equi.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with equidistant points')
fig2.savefig('Homework4/images/f1_error_equi.eps', format = 'eps')

print()
f = f1
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'cheby')
    end = perf_counter()
    print(f'Time for n = {n} and Chebyshev points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with Chebyshev points')
fig1.savefig('Homework4/images/f1_cheby.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with Chebyshev points')
fig2.savefig('Homework4/images/f1_error_cheby.eps', format = 'eps')

print()
f = f2
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'equi')
    end = perf_counter()
    print(f'Time for n = {n} and equidistant points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with equidistant points')
fig1.savefig('Homework4/images/f2_equi.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with equidistant points')
fig2.savefig('Homework4/images/f2_error_equi.eps', format = 'eps')

print()
f = f2
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'cheby')
    end = perf_counter()
    print(f'Time for n = {n} and Chebyshev points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with Chebyshev points')
fig1.savefig('Homework4/images/f2_cheby.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with Chebyshev points')
fig2.savefig('Homework4/images/f2_error_cheby.eps', format = 'eps')
        


print()

ns = [10, 20, 30]
f = f1
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'cheby')
    end = perf_counter()
    print(f'Time for n = {n} and Chebyshev points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with Chebyshev points')
fig1.savefig('Homework4/images/f3_cheby.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with Chebyshev points')
fig2.savefig('Homework4/images/f3_error_cheby.eps', format = 'eps')


print()

ns = [10, 20, 30]
f = f1
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9))
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
for i,n in enumerate(ns):
    start = perf_counter()
    p = Remez_exchange(f, 100, -1, 1, n, 4e-3, 'equi')
    end = perf_counter()
    print(f'Time for n = {n} and Equi points : {end - start}')
    ax1.plot(x_plot, p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')
    ax2.plot(x_plot, f(x_plot) - p(x_plot), linestyle=ls[i], label=f'n = {n}', color='k')


ax1.grid()
ax1.plot(x_plot, f(x_plot), 'g', label='f(x)')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Interpolants with Equi points')
fig1.savefig('Homework4/images/equi.eps', format = 'eps')

ax2.legend()
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('error')
ax2.set_title('Error with Equi points')
fig2.savefig('Homework4/images/f3_error_equi.eps', format = 'eps')


plt.show()
