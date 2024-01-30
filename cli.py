import argparse  # https://docs.python.org/3/library/argparse.html

from cnn import CNNNetwork
from inference import predict


def main():
    parser = argparse.ArgumentParser(description="Audio CNN")
    parser.add_argument("input_file", help="Path to input file.")
    parser.add_argument("output_file", help="Path to output file.")
    args = parser.parse_args()

    model = ""
    result = ""
    print(result)


if __name__ == "__main__":
    main()
