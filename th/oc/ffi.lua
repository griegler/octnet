-- Copyright (c) 2017, The OctNet authors
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--     * Redistributions of source code must retain the above copyright
--       notice, this list of conditions and the following disclaimer.
--     * Redistributions in binary form must reproduce the above copyright
--       notice, this list of conditions and the following disclaimer in the
--       documentation and/or other materials provided with the distribution.
--     * Neither the name of the <organization> nor the
--       names of its contributors may be used to endorse or promote products
--       derived from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

local ffi = require 'ffi'

--------------------------------------------------------------------------------
-- TH functions
--------------------------------------------------------------------------------
ffi.cdef[[
void* THAlloc(long size);
void THFloatStorage_free(THFloatStorage* storage);
]]


--------------------------------------------------------------------------------
-- Octree struct and types
--------------------------------------------------------------------------------
ffi.cdef[[
typedef int ot_size_t;
typedef int ot_tree_t;
typedef float ot_data_t;

typedef struct {
  ot_size_t* data;
  ot_size_t capacity;
} ot_size_t_buffer;

typedef struct {
  ot_data_t* data;
  ot_size_t capacity;
} ot_data_t_buffer;

typedef struct {
  ot_size_t n;
  ot_size_t grid_depth;
  ot_size_t grid_height;
  ot_size_t grid_width;

  ot_size_t feature_size; 

  ot_size_t n_leafs; 

  ot_tree_t* trees;      
  ot_size_t* prefix_leafs; 
  ot_data_t* data;       

  ot_size_t grid_capacity;
  ot_size_t data_capacity;
} octree;

void *malloc(size_t size);
]]

--------------------------------------------------------------------------------
-- Octree CPU
--------------------------------------------------------------------------------
ffi.cdef[[
// -----------------------------------------------------------------------------
ot_size_t_buffer* new_ot_size_t_buffer_gpu();
void free_ot_size_t_buffer_gpu(ot_size_t_buffer* buf);
void resize_ot_size_t_buffer_gpu(ot_size_t_buffer* buf, ot_size_t N);

// -----------------------------------------------------------------------------
void tree_set_bit_cpu(ot_tree_t* num, int pos);
void tree_unset_bit_cpu(ot_tree_t* num, const int pos);
bool tree_isset_bit_cpu(const ot_tree_t* num, int pos) { return tree_isset_bit(num, pos); }
int tree_n_leafs_cpu(const ot_tree_t* tree);
int tree_data_idx_cpu(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size);
int octree_mem_capacity_cpu(const octree* grid);
int octree_mem_using_cpu(const octree* grid);

int leaf_idx_to_grid_idx_cpu(const octree* grid, const int leaf_idx);
int data_idx_to_bit_idx_cpu(const ot_tree_t* tree, int data_idx);
int depth_from_bit_idx_cpu(const int bit_idx);
void octree_split_grid_idx_cpu(const octree* in, const int grid_idx, int* n, int* d, int* h, int* w);
void bdhw_from_idx_l1_cpu(const int bit_idx, int* d, int* h, int* w);
void bdhw_from_idx_l2_cpu(const int bit_idx, int* d, int* h, int* w);
void bdhw_from_idx_l3_cpu(const int bit_idx, int* d, int* h, int* w);

ot_tree_t* octree_get_tree_cpu(const octree* grid, ot_size_t grid_idx); 
void octree_clr_trees_cpu(octree* grid_h);
void octree_cpy_trees_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_cpy_prefix_leafs_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_cpy_data_cpu_cpu(const octree* src_h, octree* dst_h);
void octree_copy_cpu(const octree* src, octree* dst);
void octree_upd_n_leafs_cpu(octree* grid_h);
void octree_upd_prefix_leafs_cpu(octree* grid_h);
void octree_fill_data_cpu(octree* grid_h, ot_data_t fill_value);
void octree_cpy_sup_to_sub_cpu(const octree* sup, octree* sub);

void octree_print_cpu(const octree* grid_h);

octree* octree_new_cpu();
void octree_free_cpu(octree* grid_h);

void octree_resize_cpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);
void octree_resize_as_cpu(const octree* src, octree* dst);

void octree_read_header_cpu(const char* path, octree* grid_h);
void octree_read_cpu(const char* path, octree* grid_h);
void octree_write_cpu(const char* path, const octree* grid_h);
void octree_read_batch_cpu(int n_paths, const char** paths, int n_threads, octree* grid_h);
void octree_dhwc_write_cpu(const char* path, const octree* grid_h);
void octree_cdhw_write_cpu(const char* path, const octree* grid_h);
void dense_write_cpu(const char* path, int n_dim, const int* dims, const ot_data_t* data);
ot_data_t* dense_read_cpu(const char* path, int n_dim);
void dense_read_prealloc_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);
void dense_read_prealloc_batch_cpu(int n_paths, const char** paths, int n_threads, int n_dim, const int* dims, ot_data_t* data); 

void octree_to_dhwc_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_dhwc_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void octree_to_cdhw_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_cdhw_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void dhwc_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void dhwc_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void dhwc_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void dhwc_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void cdhw_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void cdhw_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);
void cdhw_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void cdhw_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void octree_conv3x3x3_sum_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_sum_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_sum_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);
void octree_conv3x3x3_avg_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_avg_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_avg_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_pool2x2x2_avg_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_max_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_avg_bwd_cpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);
void octree_pool2x2x2_max_bwd_cpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);

void octree_gridpool2x2x2_avg_cpu(const octree* in, octree* out);
void octree_gridpool2x2x2_max_cpu(const octree* in, octree* out);
void octree_gridpool2x2x2_avg_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridpool2x2x2_max_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);

void octree_gridunpool2x2x2_cpu(const octree* in, octree* out);
void octree_gridunpool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridunpoolguided2x2x2_cpu(const octree* in, const octree* in_struct, octree* out);
void octree_gridunpoolguided2x2x2_bwd_cpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in);

void octree_relu_cpu(const octree* grid_in, bool inplace, octree* grid_out);
void octree_relu_bwd_cpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in);
void octree_sigmoid_cpu(const octree* in, bool inplace, octree* out);
void octree_sigmoid_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in);
void octree_logsoftmax_cpu(const octree* in, octree* out);
void octree_logsoftmax_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in);

void octree_add_cpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out);
void octree_scalar_mul_cpu(octree* grid, const ot_data_t scalar);
void octree_scalar_add_cpu(octree* grid, const ot_data_t scalar);
void octree_sign_cpu(octree* grid);
void octree_abs_cpu(octree* grid);
void octree_log_cpu(octree* grid);
ot_data_t octree_min_cpu(const octree* grid_in);
ot_data_t octree_max_cpu(const octree* grid_in);

void octree_concat_cpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);
void octree_concat_bwd_cpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_dense_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);
void octree_concat_dense_bwd_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

void octree_split_by_prob_cpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out);
void octree_split_full_cpu(const octree* in, octree* out);
void octree_split_reconstruction_surface_cpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_t rec_thr_to, octree* out);
void octree_split_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_split_dense_reconstruction_surface_fres_cpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);
void octree_split_dense_reconstruction_surface_fres_bwd_cpu(const octree* grad_out, ot_data_t* grad_in);

void octree_combine_n_cpu(const octree** in, const int n, octree* out);
void octree_extract_n_cpu(const octree* in, int from, int to, octree* out);
void octree_mask_by_label_cpu(const octree* labels, int mask_label, bool check, octree* values);
void octree_determine_gt_split_cpu(const octree* struc, const ot_data_t* gt, octree* out);

ot_data_t octree_mse_loss_cpu(const octree* input, const octree* target, bool size_average, bool check);
void octree_mse_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_nll_loss_cpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_nll_loss_bwd_cpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);

void octree_bce_loss_cpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_bce_dense_loss_cpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_dense_loss_bwd_cpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);

void octree_bce_ds_loss_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_ds_loss_bwd_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);

// -----------------------------------------------------------------------------
void volumetric_nn_upsampling_cdhw_cpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out);
void volumetric_nn_upsampling_cdhw_bwd_cpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in);

// -----------------------------------------------------------------------------
THFloatStorage* octree_data_torch_cpu(octree* grid);

// -----------------------------------------------------------------------------
octree* octree_create_from_dense_cpu(const ot_data_t* data, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);
]]

--------------------------------------------------------------------------------
-- Octree GPU
--------------------------------------------------------------------------------
if cutorch then
ffi.cdef[[
// -----------------------------------------------------------------------------
ot_data_t_buffer* new_ot_data_t_buffer_gpu();
void free_ot_data_t_buffer_gpu(ot_data_t_buffer* buf);
void resize_ot_data_t_buffer_gpu(ot_data_t_buffer* buf, ot_size_t N);

// -----------------------------------------------------------------------------
cublasHandle_t octree_torch_current_cublas_handle_gpu(THCState* state);

// -----------------------------------------------------------------------------
void octree_upd_n_leafs_gpu(octree* grid_h);
void octree_upd_prefix_leafs_gpu(octree* grid_h);
void octree_cpy_trees_gpu_gpu(const octree* src_h, octree* dst_h);
void octree_cpy_prefix_leafs_gpu_gpu(const octree* src_h, octree* dst_h);
void octree_copy_gpu(const octree* src, octree* dst);
void octree_fill_data_gpu(octree* grid_h, ot_data_t fill_value);
void octree_cpy_sup_to_sub_gpu(const octree* sup, octree* sub);

octree* octree_new_gpu();
void octree_free_gpu(octree* grid_d);

void octree_to_gpu(const octree* grid_h, octree* grid_d);
void octree_to_cpu(const octree* grid_d, octree* grid_h);

void octree_resize_gpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);
void octree_resize_as_gpu(const octree* src, octree* dst);

void octree_to_dhwc_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_dhwc_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void octree_to_cdhw_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
void octree_to_cdhw_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out_data, octree* grad_grid_in_h);

void dhwc_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void dhwc_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void dhwc_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void dhwc_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void cdhw_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void cdhw_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d_out);
void cdhw_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);
void cdhw_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data);

void octree_conv3x3x3_sum_gpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_sum_bwd_gpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_sum_wbwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);
void octree_conv3x3x3_avg_gpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int out_feature_size, octree* grid);
void octree_conv3x3x3_avg_bwd_gpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);
void octree_conv3x3x3_avg_wbwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_conv_mm_gpu(cublasHandle_t cublas_handle, const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, int n_grids, octree* grid);
void octree_conv_mm_bwd_gpu(cublasHandle_t cublas_handle, const octree* grad_out, const ot_data_t* weights, int channels_in, int n_grids, octree* grad_in); 
void octree_conv_mm_wbwd_gpu(cublasHandle_t cublas_handle, const octree* in, const octree* grad_out, const float scale, int n_grids, ot_data_t* grad_weights, ot_data_t* grad_bias);

void octree_pool2x2x2_avg_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_max_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);
void octree_pool2x2x2_avg_bwd_gpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);
void octree_pool2x2x2_max_bwd_gpu(const octree* grid_in, const octree* grid_grad_out, octree* grid_grad_in);

void octree_gridpool2x2x2_avg_gpu(const octree* in, octree* out);
void octree_gridpool2x2x2_max_gpu(const octree* in, octree* out);
void octree_gridpool2x2x2_avg_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridpool2x2x2_max_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);

void octree_gridunpool2x2x2_gpu(const octree* in, octree* out);
void octree_gridunpool2x2x2_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_gridunpoolguided2x2x2_gpu(const octree* in, const octree* in_struct, octree* out);
void octree_gridunpoolguided2x2x2_bwd_gpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in);

void octree_relu_gpu(const octree* grid_in, bool inplace, octree* grid_out);
void octree_relu_bwd_gpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in);
void octree_sigmoid_gpu(const octree* in, bool inplace, octree* out);
void octree_sigmoid_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in);
void octree_logsoftmax_gpu(const octree* in, octree* out);
void octree_logsoftmax_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in);

void octree_add_gpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out);
void octree_scalar_mul_gpu(octree* grid, const ot_data_t scalar);
void octree_scalar_add_gpu(octree* grid, const ot_data_t scalar);
void octree_sign_gpu(octree* grid);
void octree_abs_gpu(octree* grid);
void octree_log_gpu(octree* grid);
ot_data_t octree_min_gpu(const octree* grid_in);
ot_data_t octree_max_gpu(const octree* grid_in);

void octree_concat_gpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);
void octree_concat_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_ds_gpu(const octree* in1, const octree* in2, octree* out);
void octree_concat_ds_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);
void octree_concat_dense_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);
void octree_concat_dense_bwd_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

void octree_split_by_prob_gpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out);
void octree_split_full_gpu(const octree* in, octree* out);
void octree_split_reconstruction_surface_gpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_t rec_thr_to, octree* out);
void octree_split_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);
void octree_split_dense_reconstruction_surface_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int structure_type, octree* out);
void octree_split_dense_reconstruction_surface_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);
void octree_split_dense_reconstruction_surface_fres_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);
void octree_split_dense_reconstruction_surface_fres_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);
void octree_split_tsdf_gpu(const ot_data_t* features, const ot_data_t* reconstruction, const octree* guide, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int band, octree* out);

void octree_mask_by_label_gpu(const octree* labels, int mask_label, bool check, octree* values);
void octree_determine_gt_split_gpu(const octree* struc, const ot_data_t* gt, octree* out);

ot_data_t octree_mse_loss_gpu(const octree* input, const octree* target, bool size_average, bool check);
void octree_mse_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);
ot_data_t octree_mse_ds_loss_gpu(const octree* input, const octree* target, bool size_average);
void octree_mse_loss_ds_bwd_gpu(const octree* input, const octree* target, bool size_average, octree* grad);

void octree_nll_loss_gpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_nll_loss_bwd_gpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);

void octree_bce_loss_gpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_bce_dense_loss_gpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_dense_loss_bwd_gpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);

void octree_bce_ds_loss_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_ds_loss_bwd_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);

void dense_bce_loss_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t* output, ot_data_t* total_weight);
void dense_bce_loss_bwd_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t total_weight, ot_data_t* grad); 

// -----------------------------------------------------------------------------
void volumetric_nn_upsampling_cdhw_gpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out);
void volumetric_nn_upsampling_cdhw_bwd_gpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in);
]]
end



--------------------------------------------------------------------------------
-- Octree load CPU and GPU lib
--------------------------------------------------------------------------------
local function script_path()
  local str = debug.getinfo(2, "S").source:sub(2)
  return str:match("(.*/)")
end
local sp = script_path()

local libnames = {sp..'../../core/build/liboctnet_core.so', sp..'../../core/build/liboctnet_core.dylib'} 
local ok = false
for i = 1, #libnames do
  ok = pcall(function () oc.cpu = ffi.load(libnames[i]) end)
  if ok then break; end
end

if not ok then
  error('[ERROR] could not find liboctnet_core')
end

local libnames = {sp..'../../th/cpu/build/liboctnet_torch_cpu.so', sp..'../../th/cpu/build/liboctnet_torch_cpu.dylib'} 
local ok = false
for i = 1, #libnames do
  ok = pcall(function () oc.torch_cpu = ffi.load(libnames[i]) end)
  if ok then break; end
end

if not ok then
  error('[ERROR] could not find liboctnet_torch_cpu')
end



if cutorch then
  local libnames = {sp..'../../core_gpu/build/liboctnet_core_gpu.so', sp..'../../core_gpu/build/liboctnet_core_gpu.dylib'} 
  local ok = false
  for i = 1, #libnames do
    ok = pcall(function () oc.gpu = ffi.load(libnames[i]) end)
    if ok then break; end
  end

  if not ok then
    error('[ERROR] could not find liboctnet_core_gpu')
  end

  local libnames = {sp..'../../th/gpu/build/liboctnet_torch_gpu.so', sp..'../../th/gpu/build/liboctnet_torch_gpu.dylib'} 
  local ok = false
  for i = 1, #libnames do
    ok = pcall(function () oc.torch_gpu = ffi.load(libnames[i]) end)
    if ok then break; end
  end

  if not ok then
    error('[ERROR] could not find liboctnet_torch_gpu')
  end

  function get_cublas_handle()
    local state = ffi.cast('THCState*', cutorch.getState())
    return oc.torch_gpu.octree_torch_current_cublas_handle_gpu(state)
  end
end
