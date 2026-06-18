# ---------------------------------------------------------------------
#
#        CREATION AND ANNIHILATION OPERATOR GENERATOR
#        FOR FERMIONIC, BOSONIC, AND JOINT FOCK SPACES
#
# ---------------------------------------------------------------------
#
# This module constructs explicit matrix representations of:
#
#   - fermionic creation and annihilation operators (d†, d)
#   - bosonic creation and annihilation operators (a†, a)
#   - combined fermion–boson tensor-product operators
#
# in finite-dimensional truncated Fock spaces.
#
# ---------------------------------------------------------------------
#
# PHYSICAL ROLE
#
# This class provides a concrete operator representation of many-body
# quantum systems in occupation-number (Fock) basis form.
#
# It is intended for constructing Hamiltonians and Liouvillians in:
#
#   - fermionic lattice / molecular electronic systems
#   - bosonic vibrational or photonic modes
#   - coupled fermion–boson hybrid systems
#
# ---------------------------------------------------------------------
#
# REPRESENTATION STRATEGY
#
# All operators are represented as explicit dense matrices acting on
# Fock-space basis states.
#
# The basis is constructed from:
#
#   - fermionic occupation-number states (antisymmetric algebra)
#   - bosonic occupation-number states (truncated harmonic oscillator)
#   - tensor products for combined systems
#
# ---------------------------------------------------------------------
#
# FERMIONIC SECTOR
#
# For m fermionic modes:
#
#   dim_el = 2^m
#
# The module constructs matrix representations of:
#
#   d_m    : annihilation operator for fermionic mode m
#   d^\dag_m   : creation operator for fermionic mode m
#
# Fermionic antisymmetry is enforced explicitly via sign tracking
# in the occupation-number basis construction.
#
# Operator storage format:
#
#   D_ops[:,:,:,0] = d†
#   D_ops[:,:,:,1] = d
#
# ---------------------------------------------------------------------
#
# BOSONIC SECTOR
#
# For Nmodes bosonic modes with truncation Nbosons:
#
# Each mode is represented in a finite harmonic oscillator basis:
#
#   a† |n> = sqrt(n+1) |n+1>
#   a  |n> = sqrt(n)   |n-1>
#
# Multimode operators are constructed via Kronecker products over modes.
#
# Operator storage format:
#
#   A_ops[:,:,:,0] = a†
#   A_ops[:,:,:,1] = a
#
# ---------------------------------------------------------------------
#
# JOINT FOCK SPACE
#
# When both fermionic and bosonic sectors are present, the total space is:
#
#   dim_rho = dim_el × Nstates_boson
#
# Operators are embedded using tensor products:
#
#   fermions \otimes identity_bosons
#   bosons   \otimes identity_fermions
#
# The resulting basis is a direct product of:
#
#   |fermionic occupation⟩ \otimes |bosonic occupation⟩
#
# ---------------------------------------------------------------------
#
# INITIALIZATION MODES
#
# The class supports three construction modes:
#
#   - "Fermi" : fermionic operators only
#   - "Bose"  : bosonic operators only
#   - "Both"  : combined fermion–boson system
#
# ---------------------------------------------------------------------
#
# OUTPUT INTERFACE
#
# Depending on initialization mode, the class provides:
#
# Fermionic:
#   D_ops, d, ddag, Fermionic_Fock_states
#
# Bosonic:
#   A_ops, a, adag, Bosonic_Fock_states
#
# Both:
#   D_ops_joint, d_joint, ddag_joint,
#   A_ops_joint, a_joint, adag_joint,
#   Both_Fock_states
#
# ---------------------------------------------------------------------
#
# NUMERICAL CHARACTERISTICS
#
# - Fully explicit dense matrix representation
# - Fermionic structure built via combinatorial enumeration
# - Bosonic structure built via truncated oscillator algebra
# - Combined spaces constructed via Kronecker products
#
# ---------------------------------------------------------------------
#
# DESIGN INTENT
#
# This implementation prioritizes:
#
#   - transparency of operator construction
#   - exact fermionic antisymmetry
#   - straightforward tensor-product embedding
#
# over computational scalability.
#
# ---------------------------------------------------------------------
#
# Note
#
# The fermionic construction explicitly enumerates occupation states
# and applies sign factors from permutation parity to enforce correct
# anticommutation relations.
#
# ---------------------------------------------------------------------

import numpy as np
import itertools
import scipy.special
from scipy.special import comb
import CreAnn
import matplotlib.pyplot as plt

class Franck_Condon():
    
    def __init__(self,Constraints,El_Ph_Int,Ph_Freq):
        
        self.Constraints = Constraints
        self.Nel = self.Constraints[0]
        self.max_ph_occ = self.Constraints[2][0]
        self.El_Ph_Int = El_Ph_Int
        self.Ph_Freq = Ph_Freq
        
        self.Laguerre_Polynomials()
        self.Franck_Condon_Matrix()
        
    def Laguerre_Polynomials(self):
        x = ((self.El_Ph_Int)**2)/((self.Ph_Freq)**2)
        self.Laguerre = np.zeros((self.max_ph_occ+1,self.max_ph_occ+1,self.Nel))
        for itrm in range(self.Nel):
            self.Laguerre[0,:,itrm] = 1
            self.Laguerre[1,:,itrm] = -x[itrm] + np.arange(self.max_ph_occ+1) + 1
            for n in range(2,self.max_ph_occ+1):
                self.Laguerre[n,:,itrm] = ((2*(n-1) + np.arange(self.max_ph_occ+1) + 1 - x[itrm])*self.Laguerre[n-1,:,itrm] - (n-1+np.arange(self.max_ph_occ+1))*self.Laguerre[n-2,:,itrm])/n

    def Franck_Condon_Matrix(self):
        dim_ph = self.max_ph_occ+1
        dim_el = 2**(self.Nel)
        dim_rho = dim_el*dim_ph
        self.FC_Matrix = np.zeros((dim_ph,dim_ph,self.Nel))
        self.FC_Matrix_Fock_Space = np.zeros((dim_rho,dim_rho,self.Nel))
        for itrm in range(self.Nel):
            for nurow in range(self.max_ph_occ+1):
                for nucol in range(self.max_ph_occ+1):
                    self.FC_Matrix[nurow,nucol,itrm] = np.exp(-0.5*((self.El_Ph_Int[itrm]/self.Ph_Freq)**2))*np.sqrt(np.math.factorial(np.amin([nurow,nucol]))/np.math.factorial(np.amax([nurow,nucol])))\
                                            *((np.sign(nucol-nurow)*(self.El_Ph_Int[itrm]/self.Ph_Freq))**(np.abs(nucol-nurow)))*self.Laguerre[np.amin([nurow,nucol]),np.abs(nucol-nurow),itrm]
            self.FC_Matrix_Fock_Space[:,:,itrm] = np.kron(self.FC_Matrix[:,:,itrm].transpose(),np.eye(dim_el))

    def return_FC_Operators(self):
        # return self.D_ops_FC,self.d_FC,self.ddag_FC,self.FC_Matrix
        return self.FC_Matrix,self.FC_Matrix_Fock_Space 

    def Dressed_FD_Functions(self,e,w,T):
        q = [np.arange(self.max_ph_occ+1)]
        FC_Sq = self.FC_Matrix[:,:,0]**2
        self.FD_01 = FC_Sq*(1 - 1/(1+np.exp((e - w*(np.transpose(q)-q))/T)))
        self.FD_10 = FC_Sq*(1/(1+np.exp((e + w*(np.transpose(q)-q))/T)))
        return self.FD_01,self.FD_10

if __name__=='__main__':
    
    import Franck_Condon

    Nel = 1
    Nph = 1
    max_ph_occ = 5    
    Constraints = np.array([Nel,Nph,[max_ph_occ]])
    El_Ph_Int = np.array([[1,3]])
    Ph_Freq = 1

    FC_Operators = Franck_Condon.Franck_Condon(Constraints,El_Ph_Int,Ph_Freq)
    # D_ops_FC,d_FC,ddag_FC,FC_Matrix = FC_Operators.return_FC_Operators()
    FC_Matrix,FC_Matrix_Fock_Space = FC_Operators.return_FC_Operators()

    FC_Operators_file = open('Franck_Condon_Operators.txt',"w")

    # FC_Operators_file.write("-----------------------------------------------------------------------------------FERMIONIC CREATION OPERATORS----------------------------------------------------------------------\n")
    # for itrm in range(Constraints[0]):
    #     np.savetxt(FC_Operators_file,ddag_FC[:,:,itrm],fmt='%-5.5f')
    #     FC_Operators_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    # FC_Operators_file.write("-----------------------------------------------------------------------------------FERMIONIC ANNIHILATION OPERATORS----------------------------------------------------------------------\n")
    # for itrm in range(Constraints[0]):
    #     np.savetxt(FC_Operators_file,d_FC[:,:,itrm],fmt='%-5.5f')
    #     FC_Operators_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    FC_Operators_file.write("-----------------------------------------------------------------------------------FRANCK-CONDON MATRICES----------------------------------------------------------------------\n")
    for itrm in range(Constraints[0]):
        np.savetxt(FC_Operators_file,FC_Matrix[:,:,itrm],fmt='%-5.5f')
        FC_Operators_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    FC_Operators_file.close()

    FC_Operators_Fock_Space_file = open('Franck_Condon_Operators_Fock_Space.txt',"w")

    FC_Operators_Fock_Space_file.write("-----------------------------------------------------------------------------------FRANCK-CONDON MATRICES----------------------------------------------------------------------\n")
    for itrm in range(Constraints[0]):
        np.savetxt(FC_Operators_Fock_Space_file,FC_Matrix_Fock_Space[:,:,itrm],fmt='%-5.5f')
        FC_Operators_Fock_Space_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    FC_Operators_Fock_Space_file.close()

    for itrm in range(Constraints[0]):
        plt.figure()
        FC_Matrix_Sq = (FC_Matrix[:,:,itrm])**2
        c = plt.imshow(FC_Matrix_Sq,cmap='inferno',extent=[0,max_ph_occ,0,max_ph_occ],origin='lower')
        plt.colorbar(c)

    plt.show()