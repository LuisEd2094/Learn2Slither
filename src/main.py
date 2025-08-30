from Python.args import get_args
from Python.learn_2_slither import Learn2Slither
from Python.menu import Menu


def main():
    args = get_args()
    if args.menu:
        menu = Menu()
        print(menu.run())
    else:
        l2s = Learn2Slither(args)
        l2s.run()


if __name__ == "__main__":
    main()
