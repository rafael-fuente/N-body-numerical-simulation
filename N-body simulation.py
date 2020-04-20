import numpy as np
import sympy as sp

# Define a Vector2D class

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Used for debugging. This method is called when you print an instance  
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) +  ")"

    def get_length(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def __add__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def __mul__(self, n):
        return Vec2(self.x * n, self.y * n)


# Define a Particle class. The particles are the bodies attracting each other

class Particle():
    # n = number of particles
    n = 0
    def __init__(self,ipos,ivel):
        
        # i = particle index
        self.i = Particle.n
        Particle.n += 1
        
        self.m = 1
        self.G = 1
        
        # pos, vel, acc = symbolic variables
        self.pos = Vec2(sp.symbols("x_"+str(self.i)),sp.symbols("y_"+str(self.i)))
        self.vel = Vec2(sp.symbols("vx_"+str(self.i)),sp.symbols("vy_"+str(self.i)))
        self.acc = Vec2(0,0)
        
        # fvel, facc = lambdify functions
        self.fvel = Vec2(None,None)
        self.facc = Vec2(None,None)
        
        # fpos, fvel = intial position and velocity
        self.ipos = ipos
        self.ivel = ivel
        
        # fnvel, fnacc = functions used in vectorfield() function
        self.fnvel = Vec2(0,0)
        self.fnacc = Vec2(0,0)
        
        # solpos, solvel = position and velocity list obtained after the integration
        self.solpos = Vec2(None,None)
        self.solvel = Vec2(None,None)
        
    def calculate_acc(self,particles):
        for j in range(len(particles)):
            if self.i !=j:
                self.acc += (particles[j].pos - self.pos)*particles[j].m*self.G*(1/(((self.pos.x-particles[j].pos.x)**2 + (self.pos.y-particles[j].pos.y)**2)**(3/2)))

    def lambdify_vel(self,particles):
        self.fvel.x = sp.lambdify(self.vel.x, self.vel.x)
        self.fvel.y = sp.lambdify(self.vel.y, self.vel.y)
   

    def lambdify_acc(self,particles):
        
        var = []
        for j in range(len(particles)):           
            var.append(particles[j].pos.x)
            var.append(particles[j].pos.y)
               
        self.facc.x = sp.lambdify([var], self.acc.x)
        self.facc.y = sp.lambdify([var], self.acc.y)







#particle list
par = []



#create the particles
par.append(Particle(Vec2(2,5),Vec2(0.5,0.5)))
par.append(Particle(Vec2(5,2),Vec2(0.5,0.2)))
par.append(Particle(Vec2(3,3),Vec2(0.1,0.5)))
par.append(Particle(Vec2(0.6,2.5),Vec2(0.5,0.5)))


n = len(par)


#create the functions to integrate
for i in range(n):
    par[i].calculate_acc(par)

for i in range(n):
    par[i].lambdify_vel(par)
    par[i].lambdify_acc(par)




import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def vectorfield(var, t):
    '''
    integrate function

    the function calculates f, a list with all differential equations of motion in the order 
    diff(x0), diff(y0), diff(x1), diff(y1)...diff(xn-1), diff(yn-1), diff(vx0), diff(vy0)...diff(vxn-1), diff(vyn-1)

    it can be optimized, but it's done to be readable
    '''
    
    pos = var[0:2*n] 
    vel = var[2*n:4*n] 
    f = []
    
    for i in range(0,n):        
        par[i].fnvel.x = par[i].fvel.x(vel[2*i])
        par[i].fnvel.y = par[i].fvel.y(vel[2*i + 1])
        f.append(par[i].fnvel.x)
        f.append(par[i].fnvel.y)
        
    for i in range(0,n):        
        par[i].fnacc.x = par[i].facc.x(pos)
        par[i].fnacc.y = par[i].facc.y(pos)
        f.append(par[i].fnacc.x)
        f.append(par[i].fnacc.y)

    return f



################################################################################################################################


from scipy.integrate import odeint


# set the initial conditions
var = []
for i in range(len(par)):
    var.append(par[i].ipos.x)
    var.append(par[i].ipos.y)
    
for i in range(len(par)):
    var.append(par[i].ivel.x)
    var.append(par[i].ivel.y)




# ODE solver parameters

tfin = 70.0
steps = 500


t = np.linspace(0,tfin,steps+1)

sol = odeint(vectorfield, var, t)
sol = np.transpose(sol)

# order the solution for clarity

for i in range(n):
    par[i].solpos.x = sol[2*i]
    par[i].solpos.y = sol[2*i+1]
    
for i in range(n):
    par[i].solvel.x = sol[2*n + 2*i]
    par[i].solvel.y = sol[2*n + 2*i+1]
    

# Calculate the total Energy of the system. The energy should be constant.

# Potential Energy
Energy = 0 
for i in range(0,n):
    for j in range(i+1,n):
        Energy += (-1/(((par[i].solpos.x-par[j].solpos.x)**2 + (par[i].solpos.y-par[j].solpos.y)**2)**(1/2)))

# Kinetic Energy
for i in range(0,n):
    Energy += 0.5*(par[i].solvel.x*par[i].solvel.x + par[i].solvel.y*par[i].solvel.y)











# Visualization of the solution with matplotlib. It use a slider to change the time

plt.style.use('dark_background')
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1,1,1)

plt.subplots_adjust(bottom=0.2,left=0.15)

ax.axis('equal')
ax.axis([-1, 30, -1, 30])
ax.set_title('Energy =' + str(Energy[0]))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)


circle = [None]*n
line  = [None]*n
for i in range(n):
    circle[i] = plt.Circle((par[i].solpos.x[0], par[i].solpos.y[0]), 0.08, ec="w", lw=2.5, zorder=20)
    ax.add_patch(circle[i])
    line[i] = ax.plot(par[i].solpos.x[:0],par[i].solpos.y[:0])[0]
    
from matplotlib.widgets import Slider

slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider = Slider(slider_ax,      # the axes object containing the slider
                  't',            # the name of the slider parameter
                  0,          # minimal value of the parameter
                  tfin,          # maximal value of the parameter
                  valinit=0,  # initial value of the parameter 
                  color = '#5c05ff' 
                 )

def update(time):
    i = int(np.rint(time*steps/tfin))
    
    ax.set_title('Energy =' + str(Energy[i]))
    for j in range(n):
        circle[j].center = par[j].solpos.x[i], par[j].solpos.y[i]
        line[j].set_xdata(par[j].solpos.x[:i+1])
        line[j].set_ydata(par[j].solpos.y[:i+1])
        
slider.on_changed(update)
plt.show()

	
