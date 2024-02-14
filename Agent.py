import numpy as np
import matplotlib.pyplot as plt
from normalize import normalize_angle
from LOS import LOS
from reward import reward_calc

class Agent() :

    def __init__(self, psii, xf, yf, xi = 0, yi = 0):
      self.term_src = ""
      self.xf = xf
      self.yf = yf
      self.x = [xi]
      self.y = [yi]
      self.psiii = psii + np.random.uniform(-np.pi,np.pi)
      self.psi = [self.psiii]
      self.delta = [0]
      self.v = 0.78
      self.r = [0]
      self.dt = 1
      self.HE, self.CTE = LOS(self.x[-1], self.y[-1],self.psi[-1], 0, 0, self.xf, self.yf)



    def dynamics(self,action):
        
        T = 1
        k = 0.145

        rudder_angle = np.deg2rad(35)*action

        r_a = self.r[-1] + self.dt*(k*rudder_angle-self.r[-1])/T
        psi_a = self.psi[-1] + self.r[-1]*self.dt
        psi_a = normalize_angle(psi_a)

        self.r.append(r_a)
        self.psi.append(psi_a)
        x_a = self.x[-1] + self.v*np.cos(self.psi[-1])*self.dt
        y_a = self.y[-1] + self.v*np.sin(self.psi[-1])*self.dt
        self.x.append(float(x_a))
        self.y.append(float(y_a))
        self.delta.append(rudder_angle)

    def obs(self):
      self.HE, self.CTE = LOS(self.x[-1], self.y[-1], self.psi[-1], 0, 0, self.xf, self.yf)
      self.observation_space = [float(self.HE), float(self.CTE),float(self.r[-1])]
      return self.observation_space

    def done_func(self):
      if np.abs(self.HE) > np.deg2rad(175):
        self.term_src = "HE"
        return True
      if np.abs(self.CTE) > 36:
        self.term_src = "CTE"
        return True
      if np.sqrt((self.xf-self.x[-1])**2 + (self.yf-self.y[-1])**2) <= 2.909:
        self.term_src = "Done"
        return True
      return False
    
    def REWARD(self):
       return reward_calc(self.HE, self.CTE, self.r[-1])


# testing
# agent = Agent(0,10,10,0,0)
# ## turning circle
# for i in range(400):
#    agent.dynamics(1)

# plt.plot(agent.x,agent.y)
# plt.show()
# xf = 10
# yf = 10
# observation = []
# agent1 = Agent(0,xf,yf,0,0)
# observation.append(agent1.obs())
# agent2 = Agent(0,xf,yf,0,0)
# observation.append(agent1.obs())
# agent3 = Agent(0,xf,yf,0,0)
# observation.append(agent1.obs())
# print(observation)
