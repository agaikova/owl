"""
Nengo Benchmark Model: Owl Auditory Localization Function

Input: Pre-established sequences of data that are intended to mimic an
experiment found in a paper investigating plasticity in the auditory localization pathway in owl.

Ouput: The numerical difference between peak values of the averaged-out tuning curves before and
after the learning supposedly happens
"""

import sys
sys.path.append('../../')

import nengo
import numpy as np
import pylab
import matplotlib.pyplot as plt

import ctn_benchmark

class Owl(ctn_benchmark.Benchmark):
    def params(self):
        self.default('time per val', time_per_val = 0.5)
        self.default('time per val testing', time_per_val_testing = 0.5)
        self.default('train interval', train_int = 45)

        self.default('childhood', childhood = 1000)
        self.default('time glasses are on', t_glasses_on = 10)

        self.default('modifier', mod = 0.401426)
        self.default('number of neurons', number_of_neurons = 200)


    def model(self, p):

        train_time = p.train_int * p.time_per_val_testing
        training1 = p.childhood + train_time
        glasses_on = training1 + p.t_glasses_on
        training2 = glasses_on + train_time

        self.training1 = training1
        self.training2 = training2
        self.glasses_on = glasses_on

        def training_fn(t):
            if (t<= p.childhood) or (training1 <= t < glasses_on):
                return 1
            else:
                return 0

        def glasses_fn(t):
            if t >= training1:
                return 1
            else:
                return 0


        dist = nengo.dists.Uniform(-np.pi/2, np.pi/2) 
        vals_testing = np.linspace(-np.pi/2,np.pi/2, p.train_int)
        training_time1 = p.childhood
        training_time2 = glasses_on-training1
        vals_training1 = dist.sample((training_time1/p.time_per_val)+1)
        vals_training2 = dist.sample((training_time2/p.time_per_val)+1)

        self.training_time1 = training_time1
        self.training_time2 = training_time2 
        self.vals_training1 = vals_training1 
        self.vals_training2 = vals_training2     

        def stim_fn(t, x):
            glasses, train = x
            if glasses < 0.5 and train >= 0.5:
                # First training period, random things appear
                index = int((t - 0)/p.time_per_val)
                return vals_training1[index% len(vals_training1)]
            if glasses < 0.5 and train < 0.5: 
                # Test the owl by forcing it to incrimentally look at all the things
                # Gives data to form pre-learning tuning curve
                index = int((t - p.childhood)/p.time_per_val_testing)
                return vals_testing[index% len(vals_testing)]
            if glasses >= 0.5 and train >= 0.5:
                # Second Training period, random things appear
                index = int((t - training1)/p.time_per_val)
                return vals_training2[index % len(vals_training2)]
            if glasses >=0.5 and train < 0.5:
                # Last period, test owl by forcing it to incrimentally look at things
                index = int((t-glasses_on)/p.time_per_val_testing)
                return vals_testing[index % len(vals_testing)]

        def convert_to_circle(x):
            #takes a value in RADIANS, converts ton x y cartesian coordinates
            # x is gonna be an angle value, we're assuming r=1 for the sake of simplicity
            return np.cos(x), np.sin(x)
            
        def is_mod_fn(t, x):
            stim, glasses = x
            if glasses >= 0.5:
                return stim+p.mod
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
            visual_cor = nengo.Ensemble(n_neurons = 200, dimensions = 2, intercepts = nengo.dists.Uniform(0.81,0.91))
            vis2vis = nengo.Connection(visual_raw[:2], visual_cor[:2],synapse = None, learning_rule_type = nengo.PES(learning_rate= 3e-4))
            vis2vis.solver = nengo.solvers.LstsqL2(weights = False)
            aud2error = nengo.Connection(auditory[:2], error[:2],transform = -1)
            vis2error = nengo.Connection(visual_cor[:2], error[:2])

            nengo.Connection(error, vis2vis.learning_rule)

            self.probe = nengo.Probe(visual_cor.neurons)
            self.probe2 = nengo.Probe(visual_raw.neurons)

        return model


    def evaluate(self, p, sim, plt):
        
        sim.run(self.training2)

        y = sim.data[self.probe]

        dt = 0.001
        mean1_array = []
        mean2_array = []
        for i in range(p.number_of_neurons):
            activity = y[:, i]
            dt = 0.001

            test1_activity = activity[int(p.childhood/dt):int(self.training1/dt)]
            test2_activity = activity[int(self.glasses_on/dt):int(self.training2/dt)]

            mean1_activity = np.mean(test1_activity.reshape(p.train_int, (p.time_per_val_testing/dt)),axis=1)
            mean2_activity = np.mean(test2_activity.reshape(p.train_int, (p.time_per_val_testing/dt)),axis=1)
            
            indices = np.where(mean1_activity==mean1_activity.max())[0]
            increment = int(max(indices))
            
            indicies_to_adjust1 = np.asarray(np.where(mean1_activity > 0))
            #indicies_to_adjust1 = np.asarray(indicies_to_adjust1)
            ind2 = np.asarray(np.where(mean2_activity>0))
            
            
            # These loops acount for ignoring the tuning curves that are cut off halfway
            if (indicies_to_adjust1[0]).size == 0 or (ind2[0]).size ==0:
                pylab.plot(mean1_activity)
                i += 1
            else:
                l = (indicies_to_adjust1[0][0])
                r = (indicies_to_adjust1[-1][-1])

                indicies_to_adjust1[:] = [(x-increment) for x in indicies_to_adjust1]

                x_axis1 = np.transpose(indicies_to_adjust1[-1])

                ############ This is the same thing but less indented and for ind2 ###########


                l2 = ind2[0][0]
                r2 = ind2[-1][-1]
                ind2[:] = [(x-increment) for x in ind2]
                x_axis2 = np.transpose(ind2[-1])

                if mean1_activity[l:r].shape != x_axis1.shape:
                    r += 1
                    if mean1_activity[l:r].shape != x_axis1.shape:
                        i +=1
                    else:
                        mean1_array.append(mean1_activity[l:r])
                        #pylab.plot(x_axis1, mean1_activity[l:r])

                if mean2_activity[l2:r2].shape != x_axis2.shape:
                    r2 +=1
                    if mean2_activity[l2:r2].shape != x_axis2.shape:
                        i +=1
                    else:
                        mean2_array.append(mean2_activity[l2:r2])
                        #pylab.plot(x_axis2, mean2_activity[l2:r2])
                elif mean1_activity[l:r].shape == x_axis1.shape and mean2_activity[l2:r2].shape == x_axis2.shape:
                    mean1_array.append(mean1_activity[l:r])
                    #pylab.plot(x_axis1, mean1_activity[l:r])
                
                    mean2_array.append(mean2_activity[l2:r2])
                    #pylab.plot(x_axis2, mean2_activity[l2:r2])
                else:
                    i+=1
                
        #print('mean1 array =', mean1_array)
        #print('mean2 array =', mean2_array)
        ###########################################################
        master = []
        master2 = []
        
        def take_average(mean_array, mean_activity, master):
            #print('Mean array SIZE', mean_array.size)
            #print('Mean activity size', mean_activity.size)
            for j in range(mean_array.size):
                cur = mean_array[j] 
                #print(cur)
                cur_length = cur.size
                bound = int(mean_activity.size/2)
                l_bound = bound * (-1)

                z = 0
                while z <= int((mean_activity.size-cur_length)/2):
                    # Add numbers to cur
                    add =  np.zeros((mean_activity.size-cur_length)/2)
                    z+=1
                
                # Force the array to be 45 things long
                cur = np.hstack((add, cur))
                cur = np.hstack((cur, add))

                peak = np.max(mean_array[j])
                peak_pt = np.max(np.where(cur == peak))
                    
                if len(cur) != 45: 
                    cur = np.hstack((0, cur))

                #print(d[peak])
                #print('length of cur', len(cur))
                #print('cur size before', cur_length)
                #print('cur size after', cur.size)
                if len(cur) == 45:
                    master.append(cur)
                    
            return master
            
        def mean(array, axis_numb):
            n = np.mean(array, axis = axis_numb)
            max_n = np.max(n)
            return(np.divide(n, max_n))

        ########################################################################
        # Graphs are gonna happen
        mean1_array = np.asarray(mean1_array)
        mean2_array = np.asarray(mean2_array)
        #print('mean1 array AGAIN =', mean1_array)
        #print('mean2 array AGAIN =', mean2_array)
        
        data1 = take_average(mean1_array, mean1_activity, master)
        data2 = take_average(mean2_array, mean2_activity, master2)
        #print(data1)
        data1 = mean(data1, 0)
        data2 = mean(data2, 0)
        #print(data1)

        x = np.where(data1 == np.max(data1))[0]
        #print(x)
        x = int(x)

        x2 = np.where(data2 == data2.max())[0]
        x2 = int(x2)
        shift = x2 - x


        pylab.plot(shift)
        return {'shift':shift}
        


if __name__ == '__main__':
    CommunicationChannel().run()
