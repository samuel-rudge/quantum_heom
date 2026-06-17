import sparse_propagation

def quantum_propagation_generator():

    def __init__(self,sparsity_object):

        self.pair_info_row_fil = sparsity_object[0]
        self.pair_info_col_fil = sparsity_object[1]
        self.nnz_elements_fil = sparsity_object[0]
        ,self.pair_values_fil,self.npairs_fil,self.npairs_uf,\
        self.nnz_elements_sparse_fil,self.nnz_elements_sparse_zeroth_tier_fil,self.row_old_indices,\
        self.atol_vec,self.rtol_vec,self.rho_nonzeros_sparse,self.isreal_sparse,self.complex_coefficients,\
        self.nnz_elements_zeroth_tier,self.rho_nonzeros,self.rho_sparsity,self.nnz_elements,
        self.is_connected_array,rho_out,self.pair_values_gamma_fil,self.si_ham_row_info_fil,\
        self.si_ham_col_info_fil,self.si_coupledown_row_info_fil,self.si_coupledown_col_info_fil,\
        self.si_coupleup_row_info_fil,self.si_coupleup_col_info_fil,\
        self.ham_loc_info_fil,self.coupleup_loc_info_fil,\
        self.coupledown_loc_info_fil,self.coupleup_conj_info_fil,self.coupledown_conj_info_fil,\
        self.pair_values_coupleup_wout_el_lead_coupling_fil,\
        self.pair_values_coupledown_wout_el_lead_coupling_fil,\
        self.trace_cols,self.rhs_vector,self.sparse_trace_array = sparsity_ingredients