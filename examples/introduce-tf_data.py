import numloopy as nplp
import loopy as lp

np = nplp.begin_computation_stack()

a = np.ones((4, 4, 4))
b = np.argument((4, 4, 4)) # indicates that this would be available later
c = 2*a + 3*b

knl, tf_data = np.end_computation_stack([c], transform=True)  # indicates to get the transform data

UNROLLED_LOOP = 0 # '0' for innermost loop, should be '2' for the outermost loop
transformed_knl = lp.tag_inames(knl,
        {tf_data[c.name][UNROLLED_LOOP]: "unr"})
print(lp.generate_code_v2(transformed_knl).device_code()) # print the generated code
