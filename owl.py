
import numpy as np
import nengo


train_int = 10
childhood = 5
training1 = childhood+train_int
glasses_on = training1+5
training2 = glasses_on+train_int
mod = 23*(np.pi/180) #degrees

def training_fn(t):
    if (t<= childhood) or (training1 <= t < glasses_on):
        return 1
    else:
        return 0

def glasses_fn(t):
    if t >= training1:
        return 1
    else:
        return 0

time_per_val = 0.5
time_per_val_testing = 1
dist = nengo.dists.Uniform(-np.pi, np.pi) 
vals_testing = np.linspace(0,2*np.pi, train_int)
training_time1 = childhood
training_time2 = glasses_on-training1
vals_training1 = dist.sample((training_time1/time_per_val)+1)
vals_training2 = dist.sample((training_time2/time_per_val)+1)
    
    
def stim_fn(t, x):
    glasses, train = x
    if glasses < 0.5 and train >= 0.5:
        # First training period, random things appear
        index = int((t - 0)/time_per_val)
        return vals_training1[index% len(vals_training1)]
    if glasses < 0.5 and train < 0.5: 
        # Test the owl by forcing it to incrimentally look at all the things
        # Gives data to form pre-learning tuning curve
        index = int((t - childhood)/time_per_val_testing)
        return vals_testing[index% len(vals_testing)]
    if glasses >= 0.5 and train >= 0.5:
        # Second Training period, random things appear
        index = int((t - training1)/time_per_val)
        return vals_training2[index % len(vals_training2)]
    if glasses >=0.5 and train < 0.5:
        # Last period, test owl by forcing it to incrimentally look at things
        index = int((t-glasses_on)/time_per_val_testing)
        return vals_testing[index % len(vals_testing)]
        
def convert_to_circle(x):
    #takes a value in RADIANS, converts ton x y cartesian coordinates
    # x is gonna be an angle value, we're assuming r=1 for the sake of simplicity
    return np.cos(x), np.sin(x)
    
def is_mod_fn(t, x):
    stim, glasses = x
    if glasses >= 0.5:
        return stim+mod
    else:
        return stim

def error_on(training):
    if training >= 0.5:
        return 1
    else:    
        return 0
        
model = nengo.Network()
with model:
    training = nengo.Node(training_fn)
    glasses = nengo.Node(glasses_fn)
    stim = nengo.Node(stim_fn, size_in = 2, size_out = 1)
    use_mod = nengo.Node(is_mod_fn, size_in = 2, size_out = 1) 
    
    train2stim = nengo.Connection(training, stim[1], synapse = None)
    glasses2stim = nengo.Connection(glasses, stim[0], synapse = None)
    
    # Connecting a bunch of different ensembles
    visual_raw = nengo.Ensemble(n_neurons=300, dimensions=3, intercepts = nengo.dists.Uniform(0.81,0.91))
    auditory = nengo.Ensemble(n_neurons=200, dimensions = 2, intercepts = nengo.dists.Uniform(0.81,0.91))
    stim2use_mod = nengo.Connection(stim, use_mod[0], synapse = None)
    glasses2use_mod = nengo.Connection(glasses, use_mod[1], synapse = None)
    use_mod2vis = nengo.Connection(use_mod, visual_raw[:2], function = convert_to_circle)
    stim2aud = nengo.Connection(stim, auditory, function = convert_to_circle)
    
    # Error stuff
    
    error = nengo.Ensemble(n_neurons = 400, dimensions = 2) #error_fn, size_in = 4, size_out = 2)
    visual_cor = nengo.Ensemble(n_neurons = 200, dimensions = 2, intercepts = nengo.dists.Uniform(-1,1))
    vis2vis = nengo.Connection(visual_raw[:2], visual_cor[:2],synapse = 0.1, learning_rule_type = nengo.PES())
    vis2vis.learning_rule_type = nengo.PES()
    nengo.Connection(error, vis2vis.learning_rule)
    vis2vis.solver = nengo.solvers.LstsqL2(weights = False)
    aud2error = nengo.Connection(auditory[:2], error[:2], transform = -1)
    vis2error = nengo.Connection(visual_cor[:2], error[:2])
    
    nengo.Connection(error, vis2vis.learning_rule)
    
    probe = nengo.Probe(visual_cor.neurons)