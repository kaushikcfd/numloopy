#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global double const *__restrict__ arr, __global double *__restrict__ arr_0)
{
  for (int i_16 = 0; i_16 <= 3; ++i_16)
    for (int i_15 = 0; i_15 <= 3; ++i_15)
    {
      arr_0[4 * i_15 + i_16] = 2.0 + arr[4 * i_15 + i_16] * 3.0;
      arr_0[16 + 4 * i_15 + i_16] = 2.0 + arr[16 + 4 * i_15 + i_16] * 3.0;
      arr_0[32 + 4 * i_15 + i_16] = 2.0 + arr[32 + 4 * i_15 + i_16] * 3.0;
      arr_0[48 + 4 * i_15 + i_16] = 2.0 + arr[48 + 4 * i_15 + i_16] * 3.0;
    }
}
