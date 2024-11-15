import numpy as np
import pandas as pd


def trying(number):
	number_exp = np.array([[1, 2, 3], [4, 5, 6]])
	df = pd.DataFrame(number_exp * number)
	return df


def main():
	trying(3)


if __name__ == '__main__':
	main()
