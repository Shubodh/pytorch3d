import numpy as np 

def align_vect(a, b):
	"""
	convert a to b
	arguments:
		a,b -> np array of shape (3,)
		(or list)
		
	returns:
		R   -> np array of shape (3,3) 
			where R in b = R@a
	"""
	a = np.array(a)
	b = np.array(b)
	if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
		print("don't use 0 vecs")
		return 
	a = a / np.linalg.norm(a)
	b = b / np.linalg.norm(b)
	v = np.cross(a, b)
	s = np.linalg.norm(v) #L2 norm
	c = np.dot(a,b)
	if c == -1:
		print("c = -1, not implemented. Vectors are opposite")
	v_x = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
	v_x_2 = np.dot(v_x, v_x)
	R = np.eye(3) + v_x + (1/(1+c)) * v_x_2
	return R

def test(a, b):
	"""
	test a, b
	arguments:
		a,b -> np array of shape (3,)
		(or list)
	"""
	a = np.array(a)
	b = np.array(b)
	print("vect a :", a)
	print("vect b :", b)
	a = np.array(a)
	b = np.array(b)
	a_hat = a / np.linalg.norm(a)
	b_hat = b / np.linalg.norm(b)
	print("unit vect a :", a_hat)
	print("unit vect b :", b_hat)
	R = align_vect(a,b)
	print("R:", R)
	print("R@a_hat", R@a_hat)
	print("b_hat", b_hat)

if __name__ == '__main__':
	print("Case 0: axis vects")
	a = [0,0,1]
	b = [1,0,0]
	test(a,b)
	print("\n\n--------\n\n")

	print("Case 1: same direction, same vect")
	a = [3,4,5]
	b = [3,4,5]
	test(a,b)
	print("\n\n--------\n\n")
	print("Case 2: specific vectors")
	a = [1,2,3]
	b = [1, 2, 1]
	test(a,b)
	print("\n\n--------\n\n")
	print("Case 3: np random")
	a = np.random.rand(3)
	b = np.random.rand(3)
	test(a,b)