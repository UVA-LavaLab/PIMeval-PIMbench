
def side1(f, x):
    return f <= x

def side2(f, x):
    # try to convert f<=x to use >

    # pre process input
    # Can't pre process input, needs to happen on pim
    #f = -1 * f  # assume we convert this for pim

    # pre process weight (amortized, so can be expensive)
    x = -1 * x 

    # pim op
    # ISSUE: how to get negative into individual cells into
    #     -> more pre processing from host, manually put each row from a mapping that says which element of the feature array needs to be negative
    #     -> can't use broadcast....
    return (-f > x) or (-f == x)


tests = [(1, 5), (6,5), (5,5), (0, 5), (10,1), (-5, -4), (-5, -6), (-5,-5), (-5, 3), (3, -5), (-6, -5), (3, 2), (5, -5)]

for test in tests:
    result = side1(test[0], test[1]) == side2(test[0], test[1])
    if result: continue  # no need to print a success
    print("--------------------------")
    print("FAILURE!!!!!")
    print(side1(test[0], test[1]), side2(test[0], test[1]))
    print("test case: ", test, " Result: ", result)