#==========================================================================================================================================
# Script written in Python to integrate the equations of motion of N particles interacting with each other gravitationally. 
# The script computes the equations of motion and use scipy.integrate to integrate them. 
# Then it uses matplotlib to visualize the solution
#==========================================================================================================================================


import numpy as np
import sympy as sp

# Define a Vector2D class

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Used for debugging. This method is called when you print an instance  
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __radd__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def __rsub__(self, v):
        return Vec2(v.x- self.x , v.y - self.y)

    def __mul__(self, n):
        return Vec2(self.x * n, self.y * n)

    def __rmul__(self, n):
        return Vec2(self.x * n, self.y * n)

    def dot(self, v):
        return self.x*v.x + self.y*v.y

    def get_length(self):
        return np.sqrt(self.dot(self) )

# Define a Particle class. The particles are the bodies attracting each other

class Particle():
    # n = number of particles
    n = 0
    def __init__(self,initial_pos,initial_vel, mass):

        # i = particle index
        self.i = Particle.n
        Particle.n += 1

        self.m = mass
        self.G = 1  # change this to 6.67408 × 1e-11 if you want real world measuring units.
        
        # pos, vel, acc = symbolic variables
        self.pos = Vec2(sp.symbols("x_"+str(self.i)),sp.symbols("y_"+str(self.i)))
        self.vel = Vec2(sp.symbols("vx_"+str(self.i)),sp.symbols("vy_"+str(self.i)))
        self.acc = Vec2(0,0)
        
        # lamb_vel, lamd_acc = lambdify functions.
        self.lamb_vel = Vec2(None,None)
        self.lamd_acc = Vec2(None,None)
        
        # fpos, lamb_vel = intial position and velocity
        self.initial_pos = initial_pos
        self.initial_vel = initial_vel
        
        # vf_vel, vf_acc = functions used in vectorfield() function
        self.vf_vel = Vec2(0,0)
        self.vf_acc = Vec2(0,0)
        
        # sol_pos, sol_vel = position and velocity solution list obtained after the integration
        self.sol_pos = Vec2(None,None)
        self.sol_vel = Vec2(None,None)
        
    # compute particle acceleration using Newton's law of universal gravitation
    def calculate_acc(self,particles):
        for j in range(len(particles)):
            if self.i !=j:
                self.acc += (particles[j].pos - self.pos)*particles[j].m*self.G*(1/(((self.pos.x-particles[j].pos.x)**2 + (self.pos.y-particles[j].pos.y)**2)**(3/2)))

    # lambdified symbolic functions are faster for numerical calculations. 
    # I used this approaach (compute first symbolic equations of motion and then compile the function with lambdify) 
    # to avoid python loops in the vectorfield function which need to be executaded thousand of times and that is slow.

    def lambdify_vel(self,particles):
        self.lamb_vel.x = sp.lambdify(self.vel.x, self.vel.x)
        self.lamb_vel.y = sp.lambdify(self.vel.y, self.vel.y)
   

    def lambdify_acc(self,particles):
        
        var = []
        for j in range(len(particles)):           
            var.append(particles[j].pos.x)
            var.append(particles[j].pos.y)
               
        self.lamd_acc.x = sp.lambdify([var], self.acc.x)
        self.lamd_acc.y = sp.lambdify([var], self.acc.y)



#Input here the initial conditions of the particles and their mass
################################################################################################################################

#particle list
par = []


#create the particles
par.append(Particle(initial_pos = Vec2(2,5), initial_vel = Vec2(0.5,0.5) , mass = 1.))
par.append(Particle(initial_pos = Vec2(5,2), initial_vel = Vec2(0.5,0.2) , mass = 1.))
par.append(Particle(initial_pos = Vec2(3,3), initial_vel = Vec2(0.1,0.5) , mass = 1.))
par.append(Particle(initial_pos = Vec2(0.6,2.5), initial_vel = Vec2(0.5,0.5) , mass = 1.))

# Simulation time and number of steps
t_end = 60.0
steps = 800


################################################################################################################################



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
        par[i].vf_vel.x = par[i].lamb_vel.x(vel[2*i])
        par[i].vf_vel.y = par[i].lamb_vel.y(vel[2*i + 1])
        f.append(par[i].vf_vel.x)
        f.append(par[i].vf_vel.y)
        
    for i in range(0,n):        
        par[i].vf_acc.x = par[i].lamd_acc.x(pos)
        par[i].vf_acc.y = par[i].lamd_acc.y(pos)
        f.append(par[i].vf_acc.x)
        f.append(par[i].vf_acc.y)

    return f




from scipy.integrate import odeint


# set the initial conditions
var = []
for i in range(len(par)):
    var.append(par[i].initial_pos.x)
    var.append(par[i].initial_pos.y)
    
for i in range(len(par)):
    var.append(par[i].initial_vel.x)
    var.append(par[i].initial_vel.y)




# ODE solver parameters


t = np.linspace(0,t_end,steps+1)

sol = odeint(vectorfield, var, t)
sol = np.transpose(sol)

# order the solution for clarity

for i in range(n):
    par[i].sol_pos.x = sol[2*i]
    par[i].sol_pos.y = sol[2*i+1]
    
for i in range(n):
    par[i].sol_vel.x = sol[2*n + 2*i]
    par[i].sol_vel.y = sol[2*n + 2*i+1]
    

# Calculate the total Energy of the system. The energy should be constant.



# Potential Energy
Energy = 0 
for i in range(0,n):
    for j in range(i+1,n):
        Energy += (-1/(((par[i].sol_pos.x-par[j].sol_pos.x)**2 + (par[i].sol_pos.y-par[j].sol_pos.y)**2)**(1/2)))

# Kinetic Energy
for i in range(0,n):
    Energy += 0.5*(par[i].sol_vel.x*par[i].sol_vel.x + par[i].sol_vel.y*par[i].sol_vel.y)






# Visualization of the solution with matplotlib. It uses a slider to change the time
################################################################################################################################


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
    circle[i] = plt.Circle((par[i].sol_pos.x[0], par[i].sol_pos.y[0]), 0.08, ec="w", lw=2.5, zorder=20)
    ax.add_patch(circle[i])
    line[i] = ax.plot(par[i].sol_pos.x[:0],par[i].sol_pos.y[:0])[0]
    
from matplotlib.widgets import Slider

slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider = Slider(slider_ax,      # the axes object containing the slider
                  't',            # the name of the slider parameter
                  0,          # minimal value of the parameter
                  t_end,          # maximal value of the parameter
                  valinit=0,  # initial value of the parameter 
                  color = '#5c05ff' 
                 )

def update(time):
    i = int(np.rint(time*steps/t_end))
    
    ax.set_title('Energy =' + str(Energy[i]))
    for j in range(n):
        circle[j].center = par[j].sol_pos.x[i], par[j].sol_pos.y[i]
        line[j].set_xdata(par[j].sol_pos.x[:i+1])
        line[j].set_ydata(par[j].sol_pos.y[:i+1])
        
slider.on_changed(update)
plt.show()

	
