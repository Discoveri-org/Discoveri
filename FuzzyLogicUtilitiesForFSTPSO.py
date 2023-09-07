import numpy as np

# Fuzzy Logic utilities, implemented following
# M. Nobile et al., Swarm and Evolutionary Computation 39 (2018) 70–85


# Membership functions
def membership_phi_worse(phi):
    phi = np.clip(phi,-1.,1.)
    return np.maximum(-phi, 0)

def membership_phi_same(phi):
    phi = np.clip(phi,-1.,1.)
    if phi <= 0:
        return np.maximum(phi + 1, 0)
    else:
        return np.maximum(1 - phi, 0)

def membership_phi_better(phi):
    phi = np.clip(phi,-1.,1.)
    return np.maximum(phi, 0)

def membership_delta_same(delta, delta1, delta2):
    if delta <= delta1:
        return 1
    elif delta1 < delta <= delta2:
        return (delta2 - delta) / (delta2 - delta1)
    else:
        return 0
        
def membership_delta_near(delta, delta1, delta2, delta3):
    if delta1 < delta <= delta2:
        return (delta - delta1) / (delta2 - delta1)
    elif delta2 < delta <= delta3:
        return 1 - (delta - delta2) / (delta3 - delta2)
    else:
        return 0

def membership_delta_far(delta, delta2, delta3, delta_max):
    if delta2 < delta <= delta3:
        return (delta - delta2) / (delta3 - delta2)
    elif delta3 < delta <= delta_max:
        return 1
    else:
        return 0
