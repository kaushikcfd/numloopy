__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global double *__restrict__ arr, __global double *__restrict__ arr_0)
{
  for (int i_4 = 0; i_4 <= 9; ++i_4)
    arr[i_4] = 1.0;
  for (int i_5 = 0; i_5 <= 9; ++i_5)
    arr_0[i_5] = arr[i_5] * 2.0 + i_5 * 3.0;
}
