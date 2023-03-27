import numpy as np

from dlgo import gotypes

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ',
    gotypes.Player.white: ' o ',
}
STR_TO_CHAR = {
    '0': ' . ',
    '-1': ' x ',
    '1': ' o ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = f"{COLS[move.point.col - 1]}{move.point.row}"
    print(f"{player} {move_str}")


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        if row <= 9:
            bump = " "
        else:
            bump = ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print(f"{bump}{row} {''.join(line)}")
    print('    ' + '  '.join(COLS[:board.num_cols]))


def point_from_coords(coords):
    """
    Transforms human input to Point object
    """
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)


def coords_from_point(point):
    return '%s%d' % (
        COLS[point.col - 1],
        point.row
    )


def print_board_from_lists(board: list[list[float]]):
    num_rows, num_cols = len(board), len(board[0])
    for row in range(num_rows-1, -1, -1):
        if row <= 8:
            bump = " "
        else:
            bump = ""
        line = []
        for col in range(num_cols):
            stone = str(int(board[row][col]))
            line.append(STR_TO_CHAR[stone])
        print(f"{bump}{row+1} {''.join(line)}")
    print('    ' + '  '.join(COLS[:num_cols]))


def transform_move_vec_to_coords(one_hot_vec: list[float], size: int = 19):
    """
    Turn one hot vector to string coordinate (ig. 21 -> P(r=3, c=4))
    """
    index = np.argmax(one_hot_vec)
    row = index // size
    col = COLS[int(index % size)]
    output = f"{col}{str(row+1)}"
    return output
