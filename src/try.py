import random
import time
import numpy as np
import pandas as pd


def trying(number):
	for i in range(int(number)):
		time.sleep(i)
		if i == 0:
			print(f'Good, you waited for {i} seconds.')
		else:
			print(f'Good, you waited for an extra {i} seconds')

	number_exp = np.array([[1, 2, 3], [4, 5, 6]])
	df = pd.DataFrame(number_exp * number)
	print(df)
	return df


def main():
	a = np.random.randint(8)
	print(f'The random number is {a}')
	trying(a)


if __name__ == '__main__':
	main()
