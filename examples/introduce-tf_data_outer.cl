#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global double const *__restrict__ arr, __global double *__restrict__ arr_0)
{
  for (int i_15 = 0; i_15 <= 3; ++i_15)
    for (int i_14 = 0; i_14 <= 3; ++i_14)
      arr_0[16 * i_14 + 4 * i_15] = 2.0 + arr[16 * i_14 + 4 * i_15] * 3.0;
  for (int i_15 = 0; i_15 <= 3; ++i_15)
    for (int i_14 = 0; i_14 <= 3; ++i_14)
      arr_0[1 + 16 * i_14 + 4 * i_15] = 2.0 + arr[1 + 16 * i_14 + 4 * i_15] * 3.0;
  for (int i_15 = 0; i_15 <= 3; ++i_15)
    for (int i_14 = 0; i_14 <= 3; ++i_14)
      arr_0[2 + 16 * i_14 + 4 * i_15] = 2.0 + arr[2 + 16 * i_14 + 4 * i_15] * 3.0;
  for (int i_15 = 0; i_15 <= 3; ++i_15)
    for (int i_14 = 0; i_14 <= 3; ++i_14)
      arr_0[3 + 16 * i_14 + 4 * i_15] = 2.0 + arr[3 + 16 * i_14 + 4 * i_15] * 3.0;
}
