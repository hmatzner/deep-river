import numpy as np
from tqdm import tqdm


def calculate_max():
    n1 = np.random.random()
    n2 = np.random.random()
    return max(n1, n2)


def calculate_square_root():
    n3 = np.random.random()
    return np.sqrt(n3)


def calculate_prob(loops):
    mx, sq = 0, 0
    for i in tqdm(range(int(loops))):
        square_root = calculate_square_root()
        maximum = calculate_max()
        if square_root > maximum:
            sq += 1
        elif maximum > square_root:
            mx += 1
    print(f'maximum was the biggest number {mx} times ({round(mx / loops * 100, 2)} % of the times)\n'
          f'square root was the biggest number {sq} times ({round(sq / loops * 100, 2)} % of the times)')


if __name__ == '__main__':
    num = 1e7
    calculate_prob(num)

