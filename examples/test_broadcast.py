import loopy as lp
import numloopy as nplp

np = nplp.begin_computation_stack()

A = np.arange(9).reshape((3, 3))
x = np.arange(3)
y = np.sum(A * x, axis=1)

knl = np.end_computation_stack([A, x, y])
print(lp.generate_code_v2(knl).device_code())
