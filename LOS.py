import numpy as np
from normalize import normalize_angle

def LOS(xc,yc,psic,xi,yi,xf,yf):
    gamma = np.arctan2((yf-yi), (xf-xi))
    cte = (yc-yi)*np.cos(gamma) - (xc-xi)*np.sin(gamma)
    delta = 2*2.902
    psi_d = gamma + np.arctan2(-cte,delta)
    psi_d = normalize_angle(psi_d)
    psi_a = normalize_angle(psic)
    he = psi_d-psi_a
    HE = normalize_angle(he)
    return HE/np.pi,cte/36



# testing
# psid = 7*np.pi/8
# psia = -np.pi/4
# x = 10*np.cos(psid)
# y = 10*np.sin(psid)
# HE,cte = LOS(0,0,psia,0,0,x,y)
# print(np.rad2deg(HE),cte)