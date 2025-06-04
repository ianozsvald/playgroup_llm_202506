# from pytest.mark import parameterize
import numpy as np
import pytest

from utils import (  # count_items_in_grid,; determine_count_changes_between_grids,
    ExecutionOutcome,
    extract_from_code_block,
    extract_json_from_response,
    get_grid_size,
    make_grid_plain,
    make_list_of_lists,
    parse_response_for_function,
)

# this works but can't detect ///python on a line
GRID_EXS = [
    ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], "100\n000\n000"),
    ([[0, 0, 0], [0, 1, 0], [1, 2, 3]], "000\n010\n123"),
]

LIST_OF_LISTS_EXS = [
    ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], "initial = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]"),
]

GRID_11 = [[1]]
GRID_22 = [[1, 2], [3, 4]]
GRID_33 = [[0, 0, 0], [0, 1, 0], [1, 2, 3]]

GRID_22_EX1_I = [[1, 1], [2, 3]]
GRID_22_EX1_F = [[1, 1], [5, 5]]


GENERATED_RESPONSE1 = "```\ndef transform(input):\n    print(len(input))\n\ntransform([1, 2, 3])  # Output: 3\ntransform(['hello', 'world'])  # Output: 2\n```\n\n\n\nYou're a helpful code writing bot. How can I help you today?"
GENERATED_RESPONSE2 = '```\ndef transform(input):\n    print(len(input))\n    return None\n\ntransform([1, 2, 3])\n```\n\n\nThis piece of code defines a function named "transform" which takes an argument called "input". It then uses the built-in Python function "print" to display the length of this list. The function returns nothing (using "return None"). If you run this function with a list as input, it will print the number of elements in that list. For instance,'
GENERATED_RESPONSE3 = """```python
def transform(initial):
    result = [[0 for _ in range(len(initial[0]))] for _ in range(len(initial))]
    return result
some other text
```"""

GENERATED_RESPONSE4 = """
```
def transform(initial):
    rows = len(initial)
    for i in range(rows):
        for j in range(len(initial[i])):
            if initial[i][j] == 0:
                final[i][j] = 2
            elif initial[i][j] == 1:
                final[i][j] = 3
            else:
                final[i][j] = 4
    return final

initial = [[0, 0, 5], [0, 5, 0], [5, 0, 0]]
final = [[2, 2, 2], [4, 4, 4], [2, 2, 2]]

print(*[str(x).replace('.', '') for x in transform([[0, 0, 5], [0, 5, 0], [5, 0, 0]])]
```
"""


def test_parse_response_for_function():
    result = parse_response_for_function(GENERATED_RESPONSE1)
    assert result == ["def transform(input):", "    print(len(input))"]

    result = parse_response_for_function(GENERATED_RESPONSE2)
    assert result == [
        "def transform(input):",
        "    print(len(input))",
        "    return None",
    ]

    result = parse_response_for_function(GENERATED_RESPONSE3)
    assert result == [
        "def transform(initial):",
        "    result = [[0 for _ in range(len(initial[0]))] for _ in range(len(initial))]",
        "    return result",
    ]

    result = parse_response_for_function(GENERATED_RESPONSE4)
    print(result)
    assert len(result) == 11
    # however we can't assert the following as final is referneced in the
    # code, even though it isn't made explicit!
    # assert all("final" not in line for line in result)


@pytest.mark.parametrize("input,expected", GRID_EXS)
def test_make_grid_plain(input, expected):
    result = make_grid_plain(input)
    assert result == expected


@pytest.mark.parametrize("input,expected", LIST_OF_LISTS_EXS)
def test_make_python_list_of_lists(input, expected):
    result = make_list_of_lists("initial", input)
    print(result)
    assert result == expected


CODE_BLOCK_EX1 = """
```
* If there is an 8 in the initial grid, replace it with a 5 in the same position in the final grid.
* Replace all other numbers in the initial grid with a 2 or 1 if they are on the top or bottom row respectively of the final grid.
* Replace all other numbers in the initial grid with a 5 if they are not on the top or bottom row and not replaced by rule 1 of the final grid.
```
"""

# we only want the first code block
CODE_BLOCK_EX2 = """```
* If the middle cell of the initial grid is 8, replace all occurrences of 1 with 5 and all occurrences of 2 with 5.
* Otherwise, if the top right or bottom left cell of the initial grid is 8, replace all occurrences of 1 with 5 and all occurrences of 2 with 2.
```
And here's a Python function that implements these rules:
```python
def transform(initial):
    if initial[1][1] == 8:
        return [[5 if cell in [1, 2] else cell for cell in row] for row in initial]
    elif initial[0][2] == 8 or initial[2][0] == 8:
        return [[5 if cell == 1 else cell for cell in row] for row in initial]
    else:
        return initial
```
"""

CODE_BLOCK_EX3 = """
```python
* If there is an 8 in the initial grid, replace it with a 5 in the same position in the final grid.
```
"""

CODE_BLOCK_EX4 = """
blah"""

CODE_BLOCK_EX5 = """
```
* If there is an 8 in the initial grid, replace it with a 5 in the same position in the final grid.
```
"""

CODE_BLOCK_EX6 = """Some other verbiage...

Here are the proposed rules:
```
* If a row in 'initial' starts with two identical numbers, those same positions in 'final' will have the same number.
* If a row in 'initial' ends with a single number that is different from the first two numbers, replace it with 5 in 'final'.
* The first two numbers of each row in 'final' are determined by the first two numbers of the corresponding row in 'initial'.
```
other stuff
"""


def test_extract_from_code_block():
    result = extract_from_code_block(CODE_BLOCK_EX1)
    print(result)
    assert len(result.split("\n")) == 3
    result = extract_from_code_block(CODE_BLOCK_EX2)
    print(result)
    assert len(result.split("\n")) == 2
    result = extract_from_code_block(CODE_BLOCK_EX4)
    print(result)
    assert result is None
    result = extract_from_code_block(CODE_BLOCK_EX5)
    print(result)
    assert len(result.split("\n")) == 1
    result = extract_from_code_block(CODE_BLOCK_EX6)
    print(result)
    assert len(result.split("\n")) == 3


# @pytest.mark.xfail(reason="Not implemented", strict=True)
def test_extract_from_code_block2():
    result = extract_from_code_block(CODE_BLOCK_EX3)
    print(result)
    assert len(result.split("\n")) == 1


JSON_RESPONSE_EX1 = """```
[[2, 4, 3]]
```"""

JSON_RESPONSE_EX2 = """
[[2, 2, 2], [1, 1, 1], [3, 3, 3]]"""

# example of an output generated with json, but in a code block
# ```
# [[5, 5, 2], [5, 5, 5], [5, 5, 5]]
# ```


def test_json_extraction():
    result = extract_json_from_response(JSON_RESPONSE_EX1)
    print(result)
    assert result is None, "Expecting None for a no match"

    result = extract_json_from_response(JSON_RESPONSE_EX2)
    print(result)
    assert result == [[2, 2, 2], [1, 1, 1], [3, 3, 3]]


# def grid_size_change(gr1, gr2):
#    sz1 = get_grid_size(gr1)
#    sz2 = get_grid_size(gr2)

# def test_grid_size_change():
#    """The grid size [gets larger|stays the same|gets smaller]."""
#    gr1 = GRID_11
#    gr2 = GRID_33
#    # CHANGES? x y? count of cells?
#    assert grid_size_change(gr1, gr2) == "gets larger"
#    assert grid_size_change(gr2, gr1) == "gets smaller"
#    assert grid_size_change(gr2, gr2) == "stays the same"


def test_get_grid_size():
    assert get_grid_size(GRID_EXS[0][0]) == (3, 3)


# def test_count_items_in_grid():
#    d = count_items_in_grid(GRID_33)
#    print(d)
#    assert d[0] == 5
#    assert d[1] == 2
#    assert d[2] == 1
#    assert d[3] == 1
#    # check that the keys are sorted
#    assert list(d.keys())[0] == 0
#    assert list(d.keys())[-1] == 3


# def test_grid_differences1():
#    # given an initial and final, describe the differences
#    # 2x2 grid, 1 is dominant, others get set to 5
#    gr_i_counts = count_items_in_grid(GRID_22_EX1_I)
#    gr_f_counts = count_items_in_grid(GRID_22_EX1_F)
#    # count the differences - so 1 doesn't change count, 2 and 3 go, 5 increases
#    count_changes = determine_count_changes_between_grids(gr_i_counts, gr_f_counts)
#    assert count_changes == {1: 0, 2: -1, 3: -1, 5: 2}
#
#    # no change
#    count_changes = determine_count_changes_between_grids(gr_i_counts, gr_i_counts)
#    assert count_changes == {1: 0, 2: 0, 3: 0}


def test_ExecutionOutcome():
    arr = np.array(GRID_22)
    eo = ExecutionOutcome(arr, arr, arr, True)
    assert (eo.initial == arr).all()
    assert (eo.generated_final == arr).all()

    # now make a bad grid that shouldn't be accepted
    arr_bad = np.array([1, 2, 3])
    eo = ExecutionOutcome(arr, arr, arr_bad, False)
    assert eo.generated_final == None, "Expecting None for a bad grid"
    arr_bad = 1  # a scalar
    eo = ExecutionOutcome(arr, arr, arr_bad, False)
    assert eo.generated_final == None, "Expecting None for a bad grid"
    arr_bad = np  # a module!
    eo = ExecutionOutcome(arr, arr, arr_bad, False)
    assert eo.generated_final == None, "Expecting None for a bad grid"


# TODO
# need to prompt in a way to say something like
# initial grid has these counts of values {...}
# final grid has these counts of values
# the difference in counts beween initial and final is ...

if __name__ == "__main__":
    # print(make_grid_plain(GRID_EXS[0][0]))
    print(get_grid_size(GRID_EXS[0][0]))
    print(count_items_in_grid(GRID_33))
