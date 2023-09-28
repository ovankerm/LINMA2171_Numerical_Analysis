from matplotlib import pyplot as plt
from math import cos

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


# ------ PLOT LAGRANGE ------
fig,ax = plt.subplots(1, 1)
ax.plot(x_plot, y_exact, 'g', label=r'$f(x)$', linewidth=linewidth)
for i in range(3):
    ax.plot(x_plot, y_plot_lagrange[i], linestyles[i], label=f'N = {N[i]}', linewidth=linewidth)
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Lagrange interpolation')
fig.savefig('images/Floater_Hormann.eps', format='eps')

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
fig.savefig('images/Lagrange.eps', format='eps')

plt.show()