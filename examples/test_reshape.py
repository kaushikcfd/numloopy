import loopy as lp
import numloopy as nplp

np = nplp.begin_computation_stack()

a = np.arange(8)
b = a.reshape((2, 4), order='F')

knl = np.end_computation_stack([a, b])
print(lp.generate_code_v2(knl).device_code())
