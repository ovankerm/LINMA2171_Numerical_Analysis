from matplotlib import pyplot as plt
from math import cos

file_suffix = "_runge_small_interval.eps"
SAVE_FLOATER = True
SAVE_LAGRANGE = True
PLOT_FIG = True
PRINT_ERRORS = True

# ------ FUNCTIONS AND CLASSES ------
def linspace(start, end, num):
    return [start + (end - start)/(num - 1) * i for i in range(0, num)]

def fact(n):
    if n == 0:
        return 1
    else :
        return fact(n-1)
    
def binom(n, k):
    return fact(n)/(fact(k) * fact(n-k))

def error(y_approx, y_exact):
    if y_approx[0] == None: return None
    N = len(y_approx)
    error_sq = [0 for i in range(N)]
    error_abs = [0 for i in range(N)]
    e_2 = 0
    e_2_norm = 0
    e_inf = 0
    e_inf_norm = 0
    for i in range(N):
        error_sq[i] = (y_approx[i] - y_exact[i])**2
        error_abs[i] = abs(y_approx[i] - y_exact[i])
        e_inf = max(e_inf, error_abs[i])
        e_inf_norm = max(e_inf_norm, y_exact[i])
        e_2 += error_sq[i]
        e_2_norm += y_exact[i]**2
    return error_sq, error_abs, (e_2/e_2_norm)**(0.5), e_inf/e_inf_norm


class Lagrange:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = len(x)
        self.factors = [0 for i in range(self.N)]
        for i in range(self.N):
            p = 1
            for j in range(self.N):
                if (i != j):
                    p *= 1/(self.x[i] - self.x[j])
            self.factors[i] = p * self.y[i]

    
    def __call__(self, x):
        s = 0
        for i in range(self.N):
            p = 1
            for j in range(self.N):
                if (i != j):
                    p *= x - self.x[j]
            s += p * self.factors[i]
        return s
        
class Floater_Hormann:
    def __init__(self, x, y, d):
        self.N = len(x)
        self.d = d
        self.x = x
        self.y = y
        self.bad = False
        if(self.d >= self.N):
            self.bad = True
        else:
            self.w = [0 for i in range(self.N)]

            for k in range(self.N):
                s = 0
                if(self.N == self.d):
                    self.w[k] = (-1)**(k-self.d) * binom(self.d, k)
                for i in range(max(0, k-self.d), min(k+1, self.N-self.d)):
                    s += binom(self.d, k-i)
                self.w[k] = s * (-1)**(k-self.d)

    def __call__(self, x):
        if(not self.bad):
            num = 0
            den = 0
            for i in range(self.N):
                temp = self.w[i]/(x - self.x[i])
                den += temp
                num += temp * self.y[i]
            return num/den
        else: return None

f1 = lambda x : cos(x)
f2 = lambda x : 1/(1 + 25 * x*x)

# ------ APPROX PARAM ------
N = [5, 10, 15]
d = [0, 3, 5, 8]
f = f2
interval = 1
x = [linspace(-interval, interval, N[0]), linspace(-interval, interval, N[1]), linspace(-interval, interval, N[2])]
y = [[0 for i in range(N[0])], [0 for i in range(N[1])], [0 for i in range(N[2])]]
for k in range(3):
    for i in range(N[k]):
        y[k][i] = f(x[k][i])

# ------ PLOTS PARAMS ------
N_plot = 1000
x_plot = linspace(-interval - 0.000001, interval + 0.000001, N_plot)
y_plot_lagrange = [[0 for i in range(len(x_plot))] for j in range(3)]
y_plot_fh = [[[0 for i in range(len(x_plot))] for j in range(3)] for k in range(4)]
y_exact = [f(x_plot[i]) for i in range(N_plot)]
linestyles = ['k-', 'k--', 'k:']
linewidth = 0.9

# ------ APPROXIMATORS ------
p1 = [Lagrange(x[i], y[i]) for i in range(3)]
p2 = [[Floater_Hormann(x[i], y[i], d[j]) for i in range(3)] for j in range(4)]
p2[2][0] = p1[0] # if n = k, polynomial interpolation

# ------ INTERPOLATION ------
for j in range(3):
    for i, x in enumerate(x_plot):
        y_plot_lagrange[j][i] = p1[j](x)
        for k in range(4):
            y_plot_fh[k][j][i] = p2[k][j](x)

# ------ ERRORS ------
errors_lagrange = [error(y_plot_lagrange[i], y_exact) for i in range(3)]
errors_fh = [[error(y_plot_fh[j][i], y_exact) for i in range(3)] for j in range(4)]

# ------ PLOT LAGRANGE ------
fig,ax = plt.subplots(1, 1, figsize=(14, 7))
ax.plot(x_plot, y_exact, 'g', label=r'$f(x)$', linewidth=linewidth)
for i in range(3):
    ax.plot(x_plot, y_plot_lagrange[i], linestyles[i], label=f'N = {N[i]}', linewidth=linewidth)
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Lagrange interpolation')
if SAVE_LAGRANGE:
    fig.savefig('Homework1/images/Lagrange'+file_suffix, format='eps')

# ------ PLOT FLOATER-HORMANN ------
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
for j in range(4):
    ax[j//2][j%2].plot(x_plot, y_exact, 'g', label=r'$f(x)$', linewidth=linewidth)
    for i in range(3):
        ax[j//2][j%2].plot(x_plot, y_plot_fh[j][i], linestyles[i], label=f'N = {N[i]}', linewidth=linewidth)
    ax[j//2][j%2].legend(loc='upper right')
    ax[j//2][j%2].grid()
    ax[j//2][j%2].set_title(f'd = {d[j]}')
    ax[j//2][j%2].set_xlabel('x')
    ax[j//2][j%2].set_ylabel('y')

fig.suptitle('Floater-Hormann interpolation', y=0.93)
if SAVE_FLOATER:
    fig.savefig('Homework1/images/Floater_Hormann'+file_suffix, format='eps')

# ------ PLOT ERRORS ------
fig,ax = plt.subplots(1, 1, figsize=(14, 7))
for i in range(3):
    ax.plot(x_plot, errors_lagrange[i][1], linestyles[i], label=f'N = {N[i]}', linewidth=linewidth)
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('error')
ax.legend()
ax.set_title('Lagrange interpolation error')
if SAVE_LAGRANGE:
    fig.savefig('Homework1/images/Lagrange_error'+file_suffix, format='eps')

fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
for j in range(4):
    for i in range(3):
        if errors_fh[j][i] != None:
            ax[j//2][j%2].plot(x_plot, errors_fh[j][i][1], linestyles[i], label=f'N = {N[i]}', linewidth=linewidth)
    ax[j//2][j%2].legend(loc='upper right')
    ax[j//2][j%2].grid()
    ax[j//2][j%2].set_title(f'd = {d[j]}')
    ax[j//2][j%2].set_xlabel('x')
    ax[j//2][j%2].set_ylabel('error')

fig.suptitle('Floater-Hormann interpolation error', y=0.93)
if SAVE_FLOATER:
    fig.savefig('Homework1/images/Floater_Hormann_error'+file_suffix, format='eps')

if PLOT_FIG:
    plt.show()

# ------ PRINT ERRORS ------
if PRINT_ERRORS:
    print("")

    print("------ Lagrange Interpolation Errors ------")
    for i in range(3):
        print(f"N = {N[i]}  ---> e_2 error : {errors_lagrange[i][2]}, e_inf error : {errors_lagrange[i][3]}")

    print("")

    print("------ Floater-Hormann Interpolation Errors ------")
    for j in range(4):
        for i in range(3):
            if errors_fh[j][i] != None:
                print(f"(N, d) = ({N[i]}, {d[j]})  ---> e_2 error : {errors_fh[j][i][2]}, e_inf error : {errors_fh[j][i][3]}")

# f(x) = cos(x)
# ------ Lagrange Interpolation Errors ------
# N = 5  ---> e_2 error : 7.091764303273428e-05, e_inf error : 0.00012705820244046979
# N = 10  ---> e_2 error : 1.2579102983679509e-09, e_inf error : 3.3957167437492626e-09
# N = 15  ---> e_2 error : 5.215117652858044e-15, e_inf error : 4.1855428998041385e-14

# ------ Floater-Hormann Interpolation Errors ------
# (N, d) = (5, 0)  ---> e_2 error : 0.0292065884816612, e_inf error : 0.046063786763806534
# (N, d) = (10, 0)  ---> e_2 error : 0.02841571269982499, e_inf error : 0.037845701507180175
# (N, d) = (15, 0)  ---> e_2 error : 0.009165929218602807, e_inf error : 0.01642730959529956
# (N, d) = (5, 3)  ---> e_2 error : 0.0018167051869256378, e_inf error : 0.0030473240870038175
# (N, d) = (10, 3)  ---> e_2 error : 0.004322801669308307, e_inf error : 0.008017342282462793
# (N, d) = (15, 3)  ---> e_2 error : 0.00019717597718646457, e_inf error : 0.0005198334421466187
# (N, d) = (5, 5)  ---> e_2 error : 7.091764303273428e-05, e_inf error : 0.00012705820244046979
# (N, d) = (10, 5)  ---> e_2 error : 0.006806875571818632, e_inf error : 0.012954562071191115
# (N, d) = (15, 5)  ---> e_2 error : 0.0002474417051622173, e_inf error : 0.0006596924868016939
# (N, d) = (10, 8)  ---> e_2 error : 0.0025461246497668357, e_inf error : 0.003472335036426721
# (N, d) = (15, 8)  ---> e_2 error : 0.0046907291166925795, e_inf error : 0.012642987812234108



# f(x) = 1/(25 * x*x)
# ------ Lagrange Interpolation Errors ------
# N = 5  ---> e_2 error : 0.7055669459089078, e_inf error : 0.43836078802986
# N = 10  ---> e_2 error : 0.27697808295736354, e_inf error : 0.30029202419926715
# N = 15  ---> e_2 error : 4.477080333444169, e_inf error : 7.192499218378267

# ------ Floater-Hormann Interpolation Errors ------
# (N, d) = (5, 0)  ---> e_2 error : 0.42280279065010107, e_inf error : 0.32036726128789667
# (N, d) = (10, 0)  ---> e_2 error : 0.1260015208818021, e_inf error : 0.16843787496552026
# (N, d) = (15, 0)  ---> e_2 error : 0.010457718599189782, e_inf error : 0.008573562129495034
# (N, d) = (5, 3)  ---> e_2 error : 0.4205159753971341, e_inf error : 0.29325432184793354
# (N, d) = (10, 3)  ---> e_2 error : 0.07771330122474897, e_inf error : 0.10768712755784186
# (N, d) = (15, 3)  ---> e_2 error : 0.019629924132430667, e_inf error : 0.014106939223517065
# (N, d) = (5, 5)  ---> e_2 error : 0.7055669459089078, e_inf error : 0.43836078802986
# (N, d) = (10, 5)  ---> e_2 error : 0.1094048759325194, e_inf error : 0.1390015992135487
# (N, d) = (15, 5)  ---> e_2 error : 0.028144719435802372, e_inf error : 0.02355663590370557
# (N, d) = (10, 8)  ---> e_2 error : 0.07970474772988058, e_inf error : 0.11190892525497112
# (N, d) = (15, 8)  ---> e_2 error : 0.016295230411251218, e_inf error : 0.01421410712077076

# On the interval [-0.15, 0.15]
# ------ Lagrange Interpolation Errors ------
# N = 10  ---> e_2 error : 6.711810284068877e-05, e_inf error : 0.00018102507342724846
# N = 30  ---> e_2 error : 7.999095601601501e-11, e_inf error : 8.498047059665013e-10
# N = 50  ---> e_2 error : 1.76915685538325e-05, e_inf error : 0.00023658454310055872