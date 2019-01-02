import argparse

def main():
    parser = argparse.ArgumentParser(description='print num')
    parser.add_argument('-n', '--number', type=int, default=5)
    args = parser.parse_args()
    number = args.number
    for i in range(number):
        print(i)

if __name__ == "__main__":
    main()
