from Python.args import get_args
from Python.learn_2_slither import Learn2Slither


def main():
    args = get_args()
    l2s = Learn2Slither(args)
    l2s.run()


if __name__ == "__main__":
    main()
