
def parse_checkerboard(checkerboard):
    return tuple(map(int, checkerboard.split("x")))


def add_common_args(arg_parser):


    arg_parser.add_argument(
        "-c", "--checkerboard",
        default='10x7',
        help="checkerboard squares layout (default '10x7')"
    )

    arg_parser.add_argument(
        "-sq", "--square-size",
        type=float,
        default=24.2,
        help="checkerboard square size [mm], (default 24.2)"
    )




