import numloopy as nplp
import loopy as lp

np = nplp.begin_computation_stack()

a = np.ones(10)
b = np.arange(10)
c = 2*a + 3*b

knl = np.end_computation_stack([a, c])
print(lp.generate_code_v2(knl.copy(target=lp.PyOpenCLTarget())).device_code())
