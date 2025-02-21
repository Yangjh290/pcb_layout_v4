from app.services.layout_service.SSA.parse_kiutils import _shape2, _shape1, _shape3
from app.services.layout_service.layout_service import _get_board_mid
from app.services.layout_service.uniform.uniform_player import _draw_board


def test_get_board():
    board_edge = _shape1()
    return _get_board_mid(board_edge, 1.5, "shape1")

if __name__ == '__main__':
    board = test_get_board()
    _draw_board(board, 1)
    print("test finished")