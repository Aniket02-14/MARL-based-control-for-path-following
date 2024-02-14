import numpy as np
import matplotlib.pyplot as  plt

def normalize_angle(theta):

  pivot  = np.sign(theta)
  if pivot >= 0:
      theta  = theta % (2*np.pi)
  else:
      theta  = theta % (-2*np.pi)

  if theta > 0 :
      if 0 < theta <= np.pi:
          theta_new  = theta
      elif theta > np.pi:
          theta_new  = theta - (2*np.pi)

  elif theta < 0:
      if 0 > theta > -np.pi:
          theta_new = theta
      elif theta < -np.pi:
          theta_new = theta + (2*np.pi)
  elif theta == 0:
      theta_new = 0
  else :
      theta_new = theta
  return theta_new


# testing
# a = []
# for i in range(0,100,1):
#     angle = normalize_angle(i/10)
#     a.append(angle)

# print(np.min(a), np.max(a))
# plt.plot(a)
# plt.show()