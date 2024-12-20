from SSA.ssa_placeutils import is_out_of_bounds
from SSA.ssa_utils import generate_queer_board
from entity.rectangle import Rectangle

if __name__ == '__main__':
    board = generate_queer_board()
    new_rect = Rectangle("test001", 41, 34, 1.4,0.7, 0)
    flag = is_out_of_bounds(new_rect, board)
    print(flag)