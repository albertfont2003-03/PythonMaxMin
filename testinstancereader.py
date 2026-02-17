from structure import instance
import os

# canvia aquest path si vols provar altres fitxers
TEST_FILES = [
    "instances/Geo/Geo 250 1.txt",
    "instances/Glover/Glover (n 15) 5.txt",
    "instances/Ran/Ran 250 6.txt"
]


def verify_instance(inst, path):

    print("\n==============================")
    print("File:", path)

    n = inst["n"]
    p = inst["p"]
    d = inst["d"]

    print("n =", n)
    print("p =", p)

    # check matrix size
    if len(d) != n:
        print("ERROR: matrix size incorrect")
        return

    # check symmetry
    ok = True
    for i in range(min(n, 10)):
        for j in range(min(n, 10)):
            if abs(d[i][j] - d[j][i]) > 1e-9:
                ok = False

    print("Matrix symmetric:", ok)

    # print sample distances
    print("Sample distances:")
    for i in range(min(3, n)):
        for j in range(min(3, n)):
            print(f"d[{i}][{j}] =", d[i][j])

    # check p validity
    if p <= 0 or p > n:
        print("ERROR: invalid p")
    else:
        print("p valid")


def main():

    for path in TEST_FILES:

        if not os.path.exists(path):
            print("File not found:", path)
            continue

        inst = instance.readInstance(path)

        verify_instance(inst, path)


if __name__ == "__main__":
    main()
