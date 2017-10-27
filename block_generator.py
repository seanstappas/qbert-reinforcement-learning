NUM_ROWS = 6
y0 = 38
x0 = 77
x_diff = 28
y_diff = 28


def generate_blocks():
    blocks = []
    for i in range(NUM_ROWS):
        for j in range(i + 1):
            new_block = y0 + 1
            blocks.append(1)
