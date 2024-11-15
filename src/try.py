import time


def trying(number):
	for i in range(int(number)):
		time.sleep(i)
		if i == 0:
			print(f'Good, you waited for {i} seconds.')
		else:
			print(f'Good, you waited for an extra {i} seconds')


def main():
	trying(4)
