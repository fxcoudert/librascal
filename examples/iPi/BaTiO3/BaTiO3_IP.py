from matplotlib import pylab as plt

import os, sys
from ase.io import read
sys.path.insert(0,"../../../build/")

import sys
import time
import rascal
import json

import ase
from ase.io import read, write
from ase.build import make_supercell
from ase.visualize import view
import numpy as np
from tqdm import tqdm_notebook
from time import time

import json

from rascal.representations import SphericalInvariants
from rascal.models import Kernel, sparse_points, train_gap_model
from rascal.neighbourlist import AtomsList
from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj, get_score, print_score

N_dataset = 1000
frames = read('BaTiO3_dataset.xyz', index=':{}'.format(N_dataset))

def extract_ref(frames,info_key='energy',array_key='zeros'):
	y,f = [], []
	for frame in frames:
		y.append(frame.info[info_key])
		if array_key is None:
			pass
		elif array_key == 'zeros':
			f.append(np.zeros(frame.get_positions().shape))
		else:
			f.append(frame.get_array(array_key))
	y= np.array(y)
	try:
		f = np.concatenate(f)
	except:
		pass
	return y,f

# Number of structures to train the model with
n = 800

global_species = []
for frame in frames:
	global_species.extend(frame.get_atomic_numbers())
global_species = np.unique(global_species)

# Split the structures in 2 sets
ids = list(range(N_dataset))
np.random.seed(10)
np.random.shuffle(ids)
np.savetxt('Structure_indices.txt', ids)

train_ids = ids[:n]

frames_train = [frames[ii] for ii in ids[:n]]

y,f = extract_ref(frames,'energy', 'forces')
y_train, f_train = extract_ref(frames_train,'energy', 'forces')

# Atomic energy baseline
atom_energy_baseline = np.mean(y)/(frames[0].get_global_number_of_atoms())
energy_baseline = {np.int(species): atom_energy_baseline for species in global_species}

# define the parameters of the spherical expansion
hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff=5.5, 
              max_radial=8, 
              max_angular=6, 
              gaussian_sigma_constant=0.5,
              gaussian_sigma_type="Constant",
              cutoff_function_type="RadialScaling",
              cutoff_smooth_width=0.5,
              cutoff_function_parameters=
                    dict(
                            rate=1,
                            scale=3.5,
                            exponent=4
                        ),
              radial_basis="GTO",
              optimization_args=
                    dict(
                            type="Spline",                                            
                            accuracy=1.0e-05                        
                        ),
              normalize=True,
              compute_gradients=True
              )


soap = SphericalInvariants(**hypers)

managers = []
for f in frames_train:
	positions = f.get_positions()
	f.set_positions(positions+[1,1,1])
	f.wrap(eps=1e-18)

start = time()
managers = soap.transform(frames_train)
print ("Execution: ", time()-start, "s")

n_sparse = {8:500, 22:250, 56:250}
compressor = CURFilter(soap, n_sparse, act_on='sample per species')
X_sparse = compressor.select_and_filter(managers)

# set up the sparse kernel and compute the representation for the train set
zeta = 4
kernel = Kernel(soap, name='GAP', zeta=zeta, target_type='Structure', kernel_type='Sparse')

# build the KNM matrix for training with forces and energies (see train_gap_model for more details)
KNM = kernel(managers, X_sparse)
KNM_grads = kernel(managers, X_sparse, grad=(True, False))
KNM = np.vstack([KNM, KNM_grads])
del KNM_grads
KNM_grads = []

# train a GAP model 
model = train_gap_model(kernel, managers, KNM, X_sparse, y_train, energy_baseline, 
                        grad_train=-f_train, lambdas=[1e-12, 1e-12], jitter=1e-13)

# save the model to a file in json format for future use
dump_obj('BaTiO3_model.json', model)
