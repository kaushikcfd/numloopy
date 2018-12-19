import faster_array as f_ray
import loopy as lp

np = f_ray.begin_computation_stack()

x = np.argument((1000, 2))
x_new = np.argument((1000, 2))

midpoint = np.sum(x, axis=0)/x.shape[0]

is_left = x[:, 0] < midpoint[0]
is_lower = x[:, 1] < midpoint[1]

is_ll = is_lower*is_left
is_ul = (1-is_lower)*is_left
is_lr = is_lower*(1-is_left)
is_ur = (1-is_lower)*(1-is_left)

num_ll = np.sum(is_ll)
num_ul = np.sum(is_ul)
num_lr = np.sum(is_lr)
num_ur = np.sum(is_ur)

idx_ll = np.cumsum(is_ll)*(is_ll)
idx_ul = (np.cumsum(is_ul) + num_ll)*(is_ul)
idx_lr = (np.cumsum(is_lr) + num_ll + num_ul)*(is_lr)
idx_ur = (np.cumsum(is_ur) + num_ll + num_ul + num_lr)*(is_ur)

indices = idx_ll + idx_ul + idx_lr + idx_ur - 1
x_new[indices, :] = x

knl = np.end_computation_stack([midpoint, is_lower, is_left,
    num_ll, num_ul, num_lr, num_ur, indices, x_new])

print(lp.generate_code_v2(knl).device_code())
