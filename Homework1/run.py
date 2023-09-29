from matplotlib import pyplot as plt
from math import cos

file_suffix = "_cos.eps"
SAVE_FIG = False
PLOT_FIG = False
PRINT_ERRORS = False

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
    return error_sq, error_abs, e_2/e_2_norm, e_inf/e_inf_norm


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
x = [linspace(-1, 1, N[0]), linspace(-1, 1, N[1]), linspace(-1, 1, N[2])]
y = [[0 for i in range(N[0])], [0 for i in range(N[1])], [0 for i in range(N[2])]]
for k in range(3):
    for i in range(N[k]):
        y[k][i] = f(x[k][i])

# ------ PLOTS PARAMS ------
N_plot = 1000
x_plot = linspace(-1.0000001, 1.0000001, N_plot)
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
if SAVE_FIG:
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
if SAVE_FIG:
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
if SAVE_FIG:
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
if SAVE_FIG:
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

    print("------ Floater-Hermann Interpolation Errors ------")
    for j in range(4):
        for i in range(3):
            if errors_fh[j][i] != None:
                print(f"(N, d) = ({N[i]}, {d[j]})  ---> e_2 error : {errors_fh[j][i][2]}, e_inf error : {errors_fh[j][i][3]}")

# f(x) = cos(x)
# ------ Lagrange Interpolation Errors ------
# N = 5  ---> e_2 error : 5.0293139021815035e-09, e_inf error : 0.0001270582001834927
# N = 10  ---> e_2 error : 1.5823387619380798e-18, e_inf error : 3.395717409880349e-09
# N = 15  ---> e_2 error : 2.5101484133139096e-29, e_inf error : 3.674840052609868e-14

# ------ Floater-Hermann Interpolation Errors ------
# (N, d) = (5, 0)  ---> e_2 error : 0.000853025117968096, e_inf error : 0.046063787008178605
# (N, d) = (10, 0)  ---> e_2 error : 0.0008074530190083818, e_inf error : 0.03784570000213539
# (N, d) = (15, 0)  ---> e_2 error : 8.401428856993519e-05, e_inf error : 0.01642730581980803
# (N, d) = (5, 3)  ---> e_2 error : 3.3004189242267272e-06, e_inf error : 0.003047324184469707
# (N, d) = (10, 3)  ---> e_2 error : 1.8686620981159516e-05, e_inf error : 0.008017343128908118
# (N, d) = (15, 3)  ---> e_2 error : 3.887837969840777e-08, e_inf error : 0.0005198333202118483
# (N, d) = (5, 5)  ---> e_2 error : 5.0293139021815035e-09, e_inf error : 0.0001270582001834927
# (N, d) = (10, 5)  ---> e_2 error : 4.633357168351652e-05, e_inf error : 0.012954563669858645
# (N, d) = (15, 5)  ---> e_2 error : 6.122741905468128e-08, e_inf error : 0.0006596923502273893
# (N, d) = (10, 8)  ---> e_2 error : 6.482753063139127e-06, e_inf error : 0.003472334918531723
# (N, d) = (15, 8)  ---> e_2 error : 2.2002947418139367e-05, e_inf error : 0.012642986971682979



# f(x) = 1/(25 * x*x)
# ------ Lagrange Interpolation Errors ------
# N = 5  ---> e_2 error : 0.49782471929229927, e_inf error : 0.4383607770453408
# N = 10  ---> e_2 error : 0.07671685870683417, e_inf error : 0.3002920615239576
# N = 15  ---> e_2 error : 20.044247658645283, e_inf error : 7.192503932487124

# ------ Floater-Hermann Interpolation Errors ------
# (N, d) = (5, 0)  ---> e_2 error : 0.1787622012949484, e_inf error : 0.320367260147341
# (N, d) = (10, 0)  ---> e_2 error : 0.015876383399105888, e_inf error : 0.16843787499409374
# (N, d) = (15, 0)  ---> e_2 error : 0.00010936387920716343, e_inf error : 0.008573561688382622
# (N, d) = (5, 3)  ---> e_2 error : 0.1768336870552963, e_inf error : 0.29325432035482735
# (N, d) = (10, 3)  ---> e_2 error : 0.006039357238251586, e_inf error : 0.10768712757754066
# (N, d) = (15, 3)  ---> e_2 error : 0.0003853339239288092, e_inf error : 0.014106937610770033
# (N, d) = (5, 5)  ---> e_2 error : 0.49782471929229927, e_inf error : 0.4383607770453408
# (N, d) = (10, 5)  ---> e_2 error : 0.011969426978701654, e_inf error : 0.1390015992377249
# (N, d) = (15, 5)  ---> e_2 error : 0.0007921252366813573, e_inf error : 0.0235566336285566
# (N, d) = (10, 8)  ---> e_2 error : 0.0063528468644932995, e_inf error : 0.11190892527529404
# (N, d) = (15, 8)  ---> e_2 error : 0.00026553453567370873, e_inf error : 0.014214105780099665