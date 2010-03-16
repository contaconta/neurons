#ifndef GPU_COMMON_H_
#define GPU_COMMON_H_

////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 16
#define   ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 3
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 3
#define   DEPTH_BLOCKDIM_Y 16
#define   DEPTH_BLOCKDIM_Z 16
#define   DEPTH_RESULT_STEPS 4
#define   DEPTH_HALO_STEPS 3

#define   SVM_BLOCKDIM_X 10
#define   SVM_BLOCKDIM_Y 10


#endif
