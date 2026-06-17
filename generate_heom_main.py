# ---------------------------------------------------------------------
#
#                     GENERARTING X-DEPENDENT HEOM CLASS
#
# ---------------------------------------------------------------------
#
# This Python file contains the generate the HEOM superoperator L_HEOM as a function 
# of vibrational coordinate x. Because implementing HEOM is a complex task, various parts 
# of the implementation are split into Python modules and Fortran
# subroutines. This code imports all modules and runs them in the correct order.
#
# USAGE - RUN FROM COMMAND LINE (TERMINAL) WITH ANACONDA:
#       python generate_heom_main.py
#
#       Note that one must create the Python wrappers from the Fortran subroutines first (eta_gamma,sparsity,sparse_propagation)
#       Alternatively, one could go into a python environment (type Python into command line) and run each line manually.
#

#### IMPORT PYTHON MODULES ####

from re import S
from shutil import make_archive
from tokenize import Double
import numpy as np                                                                                          # Import intrinsic Python modules
import scipy as sc
from time import perf_counter, asctime
import pickle,sys,gc,os
from importlib import reload
from matplotlib import pyplot as plt

#### IMPORT HEOM MODULES ####

from constants import *                                                                                     # Import HEOM modules
import system
from input_parameters import *
import Index_pm_filter
import generating_sparsity_class
import eta_gamma_barycentric
import gmres_ss_solver
import generate_heom_one_x
import sparse_propagation

redo_simulation_info = True
redo_sparsity = True
redo_adiabatic_results = True
gc.collect()

t_start_program = asctime()                     
if os.path.isfile('simulation_info.dat') and not redo_simulation_info:
    output_info = open('simulation_info.dat','a')
else:
    output_info = open('simulation_info.dat','w') 
    output_info.write('The simulation starts at '+str(t_start_program)+'\n')

# ---------------------------------------------------------------
#            DEFINE OPERATORS IN MOLECULAR HILBERT SPACE
# ---------------------------------------------------------------

system_output = system.system_operators(Single_El_Int,Double_El_Int,Nel,N_qu_vib_modes,El_Nuclear_Couplings_cl,
                                        max_occ_qu_vib_modes,dim_rho)

d_ops,d,ddag,Fock_states,molham_func,molham_log,d_ops_log,rho_0_log,identity_dim_rho,el_occ_op,el_occ_op_log = system_output[0:11]
if bool(N_qu_vib_modes):
    b_ops,b,bdag = system_output[11:14]

# ---------------------------------------------------------------
#      BATH-CORRELATION EXPANSION - BARYCENTRIC AND PADE
# ---------------------------------------------------------------

EtaGamma = eta_gamma_barycentric.bath_correlation_decomposition(Ncutoff,specwidth,Nsupport_points_barycentric,
                Npoles_pade,symmetrized_fermi_specwidth,Temp,Nleads,Nsign,muvec,tol_Gamma_barycentric,
                tol_fermi_symmetrized_barycentric,wbl_YN,analytic_spectral_function_decomposition,tol_F)
eta_vec_barycentric,gamma_vec_barycentric = EtaGamma.barycentric_bath_correlation_expansion()
eta_vec_pade,gamma_vec_pade = EtaGamma.pade_bath_correlation_expansion()

if pole_choice == "pade":
    eta_vec = eta_vec_pade
    gamma_vec = gamma_vec_pade
elif pole_choice == "barycentric":
    eta_vec = eta_vec_barycentric
    gamma_vec = gamma_vec_barycentric
elif pole_choice == "prony":
    raise ValueError("Prony/MPM decomposition not yet implemented")
    # eta_vec = eta_vec_prony
    # gamma_vec = gamma_vec_prony
else:
    raise ValueError("Choose an appropriate pole decomposition scheme: Options are pade or barycentric")

if wbl_YN == 0:                                                                    # If not under the wide-band limit, they are assumed to have a Lorentzian density of states
    Npoles = len(eta_vec[0,0,:]) - 1
    Nmodes = (Npoles+1)*Nel*Nleads*Nsign                                           # Calculate number of modes outside of wide-band limit
elif wbl_YN == 1:
    Npoles = len(eta_vec[0,0,:])
    Nmodes = Npoles*Nel*Nleads*Nsign         

sys.stdout.flush()
t_start = perf_counter()                                                                               # Return value of performance counter (internal clock with no reference time)

# ---------------------------------------------------------------
#                 INDEX GENERATION OF ADOs IN HEOM
# ---------------------------------------------------------------

Indices = Index_pm_filter.Hierarchy_index(Nmax,Nel,Npoles,Nleads,Nsign,Nmodes,wbl_YN)                              # Define object of Hierarchy_index class with HEOM parameters as input
if wbl_YN == 0:
    if filtering_YN == 1:                                                                                   # Run filtering process if filtering_YN is true
        max_V_km = V_Km
        KsigLm_filtered,Un_Ind_filtered,Hier_ind_filtered,Index_minus_filtered,Index_plus_filtered = Indices.Print_Filtered_Ind_Info(tol,eta_vec,gamma_vec,max_V_km)
    else:
        KsigLm,Un_Ind,Hier_ind,Index_Minus,Index_Plus,len_un_ind,len_index_plus,tier_index = Indices.Print_Ind_Info()
                                                                                                            # Return index information from Indices object; see Index_pm_filter for details
elif wbl_YN == 1:
    if filtering_YN == 1:                                                                                   # Run filtering process if filtering_YN is true
        max_V_km = V_Km                                                                        # Define maximum coupling between leads and electronic levels in the system
        KsigLm_filtered,Ksig0m_filtered,Un_Ind_filtered,Hier_ind_filtered,Index_minus_filtered,Index_plus_filtered = Indices.Print_Filtered_Ind_Info(tol,eta_vec,gamma_vec,max_V_km)
    else:
        KsigLm,Ksig0m,Un_Ind,Hier_ind,Index_Minus,Index_Plus,len_un_ind,len_index_plus,tier_index = Indices.Print_Ind_Info()
                                                                                                            # Return index information from Indices object; see Index_pm_filter for details

sys.stdout.flush()
t_end = perf_counter()                                                                                      # Return value of performance counter at end of index generation
output_info.write("Elapsed time of indices generation: " + str(t_end-t_start) +'\n')                        # Write into simulation_info.txt the time taken to perform index generation
output_info.close()
output_info = open('simulation_info.dat','a')

# ---------------------------------------------------------------
#          TRANSFORMING HEOM TO SPARSE REPRESENTATION 
# ---------------------------------------------------------------

sparsity_key = True
if not redo_everything:
    if os.path.isfile("sparsity_ingredients.p"):
        if not bool(redo_sparsity):
            sparsity_key = False

sparsity_key = True
if sparsity_key:
    x_vec_tester = np.array([0])
    sparsity_object = generating_sparsity_class.sparsity_heom_liouvillian(ksiglm=KsigLm,tier_index=tier_index,
                            index_minus=Index_Minus,index_plus=Index_Plus,d_ops_comp=d_ops,
                            d_ops_comp_log=d_ops_log,ham_log=molham_log,rho_0_log=rho_0_log,max_expan_order=max_expan_order,
                            dim_rho=dim_rho,len_index_plus=len_index_plus,len_un_ind=len_un_ind,nmax=Nmax,nel=Nel,
                            degenerate_levels=degenerate_levels,atol=atol,rtol=rtol,un_ind=Un_Ind,
                            gamma_vec=gamma_vec,eta_vec=eta_vec,nsign=Nsign,nleads=Nleads,npoles=Npoles,
                            molham_one_x=molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_vec_tester),
                            el_lead_couplings_one_x=el_lead_couplings_func(Nleads,Nel,V_Km,x_vec_tester))
    sparsity_object.save_sparse_heom()
    sparsity_ingredients_file = open("sparsity_ingredients.p","rb")
    sparsity_ingredients = pickle.load(sparsity_ingredients_file)
    sparsity_ingredients_file.close()
    print("Sparsity has been (re)generated")
else:
    sparsity_ingredients_file = open("sparsity_ingredients.p","rb")
    sparsity_ingredients = pickle.load(sparsity_ingredients_file)
    sparsity_ingredients_file.close()
    print("Sparsity has been loaded and not regenerated")

sys.stdout.flush()

### Generate sparse information on Liouvillian of HEOM ###

pair_info_row_fil,pair_info_col_fil,pair_values_fil,npairs_fil,npairs_uf,\
nnz_elements_sparse_fil,nnz_elements_sparse_zeroth_tier_fil,row_old_indices,\
atol_vec,rtol_vec,rho_nonzeros_sparse,isreal_sparse,complex_coefficients,\
nnz_elements_zeroth_tier,rho_nonzeros,rho_sparsity,nnz_elements,is_connected_array,rho_out,\
pair_values_gamma_fil,si_ham_row_info_fil,si_ham_col_info_fil,si_coupledown_row_info_fil,\
si_coupledown_col_info_fil,si_coupleup_row_info_fil,si_coupleup_col_info_fil,\
ham_loc_info_fil,coupleup_loc_info_fil,\
coupledown_loc_info_fil,coupleup_conj_info_fil,coupledown_conj_info_fil,\
pair_values_coupleup_wout_el_lead_coupling_fil,\
pair_values_coupledown_wout_el_lead_coupling_fil,\
trace_cols,rhs_vector,sparse_trace_array = sparsity_ingredients

t_start = t_end
t_end = perf_counter()
output_info.write("Elapsed time to transform to sparse representation: " + str(t_end-t_start) +'\n')
output_info.close()
output_info = open('simulation_info.dat','a')

# --------------------------------------------------------------------------
#               STEADY STATE VIA DIRECT SOLVER WITH GMRES  
# --------------------------------------------------------------------------

x_vec = np.array([-5],dtype=float)
ham_x = molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_vec)
el_lead_couplings_x = el_lead_couplings_func(Nleads,Nel,V_Km,x_vec)

# reload(gmres_ss_solver)
# adiabatic_ss_object = gmres_ss_solver.steady_state_x_grid(pair_info_col=pair_info_col_fil,
#     pair_info_row=pair_info_row_fil,pair_values=pair_values_fil,npairs=npairs_fil,
#     nnz_elements_sparse=nnz_elements_sparse_fil,sparse_trace_array=sparse_trace_array,
#     rhs_vector=rhs_vector,nnz_elements_sparse_zeroth_tier=nnz_elements_sparse_zeroth_tier_fil,
#     complex_coefficients=complex_coefficients,d_ops=d_ops,isreal_sparse=isreal_sparse,
#     rho_nonzeros_sparse=rho_nonzeros_sparse,tier_index=tier_index,un_ind=Un_Ind,ksiglm=KsigLm,ham_x=ham_x,
#     trace_cols=trace_cols,fock_states=Fock_states,el_occ=el_occ_op,el_occ_log=el_occ_op_log,
#     el_lead_couplings_x=el_lead_couplings_x,x_vec=x_vec_tester)
# adiabatic_ss_results = adiabatic_ss_object.return_ss_all_x_values()
# pickle.dump(adiabatic_ss_results,open("adiabatic_ss_results.p","wb"))
# print("Adiabatic steady state has been (re)generated")
# rho_ss_x_arr,rho_system,current_vec,el_occ_vec = adiabatic_ss_results
# print(rho_system)
# print(np.sum(np.diag(rho_system)))

t_start = t_end                                                                                             # Define start time of time-propagation code
t_end = perf_counter()                                                                                      # Define end time of time-propagation code
output_info.write("Elapsed time after calculating adiabatic quantities via GMRES: " + str(t_end-t_start) +'\n')                               # Write the total elapsed computer time for time-propagation code to run
output_info.close()
output_info = open('simulation_info.dat','a')

pair_values_one_x = generate_heom_one_x.heom_liouvillian_one_x(pair_values_gamma=pair_values_gamma_fil,
                                si_ham_row_info=si_ham_row_info_fil,si_ham_col_info=si_ham_col_info_fil,
                                si_coupledown_row_info=si_coupledown_row_info_fil,
                                si_coupledown_col_info=si_coupledown_col_info_fil,
                                si_coupleup_row_info=si_coupleup_row_info_fil,
                                si_coupleup_col_info=si_coupleup_col_info_fil,
                                ham_loc_info=ham_loc_info_fil,coupleup_loc_info=coupleup_loc_info_fil,
                                coupledown_loc_info=coupledown_loc_info_fil,coupleup_conj_info=coupleup_conj_info_fil,
                                coupledown_conj_info=coupledown_conj_info_fil,
                                pair_values_coupleup_wout_el_lead_coupling=pair_values_coupleup_wout_el_lead_coupling_fil,
                                pair_values_coupledown_wout_el_lead_coupling=pair_values_coupledown_wout_el_lead_coupling_fil,
                                ham_x=ham_x,el_lead_couplings_x=el_lead_couplings_x,npairs=npairs_fil,
                                nleads=Nleads,nel=Nel,dim_rho=dim_rho)

t_start = t_end
t_end = perf_counter()
output_info.write("Elapsed time to transform to sparse representation WITH SPEED: " + str(t_end-t_start) +'\n')
output_info.close()
output_info = open('simulation_info.dat','a')

adiabatic_ss_object = gmres_ss_solver.steady_state_x_grid(pair_info_col=pair_info_col_fil,
    pair_info_row=pair_info_row_fil,pair_values=pair_values_one_x,npairs=npairs_fil,
    nnz_elements_sparse=nnz_elements_sparse_fil,sparse_trace_array=sparse_trace_array,
    rhs_vector=rhs_vector,nnz_elements_sparse_zeroth_tier=nnz_elements_sparse_zeroth_tier_fil,
    complex_coefficients=complex_coefficients,d_ops=d_ops,isreal_sparse=isreal_sparse,
    rho_nonzeros_sparse=rho_nonzeros_sparse,tier_index=tier_index,un_ind=Un_Ind,ksiglm=KsigLm,ham_x=ham_x,
    trace_cols=trace_cols,fock_states=Fock_states,el_occ=el_occ_op,el_occ_log=el_occ_op_log,
    el_lead_couplings_x=el_lead_couplings_x,x_vec=x_vec)
adiabatic_ss_results = adiabatic_ss_object.return_ss_all_x_values()
rho_ss_x_arr,rho_system,current_vec,el_occ_vec = adiabatic_ss_results

t_end = asctime()                                                                                           
output_info.write('The HEOM generation ends at '+str(t_end)+'\n')
output_info.close()  

max_time = 1000
dt_init = 0.01
nsteps = int(max_time/dt_init)
time_vec = np.linspace(0,max_time,nsteps)
rho_input = np.zeros(nnz_elements_sparse_fil,dtype=float) ; rho_input[0] = 0.5 ; rho_input[1] = 0.5
rho_mol = np.zeros((nsteps,2),dtype=float)
rho_deriv = np.zeros(nnz_elements_sparse_fil,dtype=float)
rho_temp = np.zeros(nnz_elements_sparse_fil,dtype=float)
for itrt in range(len(time_vec)):
    rho_output = sparse_propagation.sparse_one_step_propagation(pair_info_row=pair_info_row_fil,
                    pair_info_col=pair_info_col_fil,pair_values=pair_values_one_x,dt=dt_init,
                    rho_input=rho_input,max_expan_order=max_expan_order,nthreads_liouvillian=nthreads_liouvillian,
                    npairs=npairs_fil,nnz_elements=nnz_elements_sparse_fil,rk_coeff=rk_coeff,rho_temp=rho_temp,
                    rho_deriv=rho_deriv)
    rho_mol[itrt,:] = rho_output[0:2]
    rho_input = rho_output

plt.plot(time_vec,rho_mol[:,0]) ; plt.show()
print(rho_output[0:2])
