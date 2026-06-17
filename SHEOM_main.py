# ---------------------------------------------------------------------
#
#        DEFINING HEOM AND DOING QUANTUM PROPAGATION EXAMPLE MAIN
#
# ---------------------------------------------------------------------
#
# This Python file generates the quantum HEOM and demonstrates how to use it in a 
# dynamical way within a time propagation. 
# 
# Note that one must create the Python wrappers from the Fortran subroutines first 
# (eta_gamma,sparsity,sparse_propagation). These can be run from the command line as 
# ./compile_f2py.sh
#
# There are no direct inputs, rather, one must first change the input_parameters.py and 
# system.py file to reflect the problem you want to solve. These
# are imported automatically into this code. 
#
# USAGE - :
#
#       python3 SHEOM_main.py
#
# OUTPUT -
#
#       At the moment, there is no output 
#

import generating_quantum_heom_class
import generate_heom_one_x
import sparse_propagation
import calculate_quantum_observables
from importlib import reload
from input_parameters import *

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
import numpy as np

### GENERATE QUANTUM HEOM INGREDIENTS ###

quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
sparse_heom_ingredients = quantum_heom_ingredients_object.return_sparse_heom_ingredients()
molecular_system_ingredients = quantum_heom_ingredients_object.return_molecular_system_ingredients()
quantum_observables_object = calculate_quantum_observables.quantum_observables_class(sparse_heom_ingredients,
                                                                                molecular_system_ingredients)

### COLLECT/DEFINE NECESSARY INGREDIENTS FOR QUANTUM HEOM PROPAGATION ###

molham_func = molecular_system_ingredients[4] # Function taking vibrational coordinates as input and returning 
                                              # H_mol as output
d_ops = molecular_system_ingredients[0] # Generate fermionic ann./cre. operators in molecular Hilbert space
d_ops_dressed = molecular_system_ingredients[9]
Ham = molecular_system_ingredients[11]
Ham_log = molecular_system_ingredients[12]
pair_info_row_fil = sparse_heom_ingredients[0]        # Rows of quantum HEOM superoperator containing nonzero elements
pair_info_col_fil = sparse_heom_ingredients[1]        # Columns of quantum HEOM superoperator containing nonzero elements
pair_values_fil = sparse_heom_ingredients[2]
npairs_fil = sparse_heom_ingredients[3]               # Number of filled elements in quantum HEOM superoperator
nnz_elements_sparse_fil = sparse_heom_ingredients[5]        # Number of ADO elements coupled to dynamics
rho_nonzeros_sparse = sparse_heom_ingredients[10]

rho_deriv = np.zeros(nnz_elements_sparse_fil,dtype=float) # Two arrays necessary for Runge-Kutta algorithm
rho_temp = np.zeros(nnz_elements_sparse_fil,dtype=float)
rho_ic = np.array([[0,0],[0,1]],dtype=float)
rho_input = np.zeros(nnz_elements_sparse_fil,dtype=float) ; rho_input[0] = rho_ic[0,0] ; rho_input[21] = rho_ic[1,1]
                                                          # Define initial condition of molecular system
rho_mol = np.zeros((n_timesteps,dim_rho),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                # with \rho_00(t) and \rho_11(t)
x_vec = np.zeros((N_cl_vib_modes,n_timesteps),dtype=float)          # Define initial array of vibrational coordinates
p_vec = np.array((N_cl_vib_modes,n_timesteps),dtype=float)          # Define initial array of vibrational momenta

mol_pops = np.zeros((n_timesteps,dim_rho),dtype=float)
current = np.zeros((n_timesteps,Nleads),dtype=float)
quantum_observables_ic = quantum_observables_object.return_quantum_observables_this_x(rho_input,el_lead_couplings_func(Nleads,Nel,V_Km,x_vec))
mol_pops[0,:] = quantum_observables_ic[2]
current[0,:] = quantum_observables_ic[0]


### TIME-PROPAGATION EXAMPLE ###
percentage_completed_old = 0
for itrt in range(n_timesteps):
    ### GENERATE MOLECULAR SYSTEM HAMILTONIAN AND MOL-METAL COUPLING AT THIS VIBRATIONAL COORDINATE
    ham_this_x = Ham
    # molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_vec[:,itrt]) 
    #                                     # Return mol. Hamiltonian at this vibrational coordinate
    el_lead_couplings_this_x = el_lead_couplings_func(Nleads,Nel,V_Km,x_vec[:,itrt]) 
                                        # Return molecule-metal coupling at this point
    ### GENERATE HEOM AT THIS VIBRATIONAL COORDINATE ###
    pair_values_this_x = pair_values_fil
    # quantum_heom_ingredients_object.return_sparse_heom_one_x(ham_this_x,
                                                                                # el_lead_couplings_this_x)
                                        # This "basically" returns the values of the nonzero elements
                                        # of the HEOM Liouvillian at this vibrational coordinate. 
    ### DO QUANTUM PART OF PROPAGATION ###
    rho_output = sparse_propagation.sparse_one_step_propagation(pair_info_row=pair_info_row_fil,
                    pair_info_col=pair_info_col_fil,pair_values=pair_values_this_x,dt=dt_init,
                    rho_input=rho_input,max_expan_order=max_expan_order,nthreads_liouvillian=nthreads_liouvillian,
                    npairs=npairs_fil,nnz_elements=nnz_elements_sparse_fil,rk_coeff=rk_coeff,rho_temp=rho_temp,
                    rho_deriv=rho_deriv) # Run one timestep of fourth-order Runge-Kutta HEOM propagation.
                                         # rho_output contains rho_mol + all ADOs at this timestep
    ### OBTAIN QUANTUM OBSERVABLES FOR THIS VIBRATIONAL COORDINATE ###
    quantum_observables = quantum_observables_object.return_quantum_observables_this_x(rho_output,el_lead_couplings_this_x)
    mol_pops[itrt,:] = quantum_observables[2]
    current[itrt,:] = quantum_observables[0]
    ### UPDATE VIBRATIONAL DEGREES OF FREEDOM ###
    # We need to fill this part in
    rho_input = rho_output # Set up for next iteration of loop
    percentage_completed = 100*itrt/n_timesteps
    if percentage_completed > (percentage_completed_old + 1):
        print("Completed ",percentage_completed,"%")
        percentage_completed_old = percentage_completed
        sys.stdout.flush()

np.savetxt("mol_pops.dat",mol_pops)
np.savetxt("current.dat",current)

mol_pops = np.genfromtxt("mol_pops.dat")
current = np.genfromtxt("current.dat")

el_pops = np.zeros((n_timesteps,dim_el),dtype=float)
el_pops[:,0] = np.sum(mol_pops[:,::2],axis=1)
el_pops[:,1] = np.sum(mol_pops[:,1::2],axis=1)
vib_pops = mol_pops[:,::2] + mol_pops[:,1::2]
n_vib = np.sum(vib_pops[:,:]*np.arange(max_occ_qu_vib_modes[0]+1),axis=1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', linewidth=2)
plt.rc('text.latex', preamble=r'\boldmath')
fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \rho_{ii}$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \mbox{\textbf{Time}} \: [\omega t] $",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_qu[0]*time_vec,el_pops[:,0],color='red',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_qu[0]*time_vec,el_pops[:,1],color='blue',linestyle='-',linewidth=2)

occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
                    Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
# ax.set_xlim(0,Vib_Freq_qu[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle E_{\mbox{\textbf{vib.}}}$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \mbox{\textbf{Time}} \: [\omega t] $",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_qu[0]*time_vec,Vib_Freq_qu[0]*(n_vib+0.5)/Temp,"b",linestyle='-',linewidth=2)
# ax.plot(time_vec,el_pops[:,1],color='blue',linestyle='-',linewidth=2)

occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
                    Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
ax.set_xlim(0,Vib_Freq_qu[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
plt.show()
