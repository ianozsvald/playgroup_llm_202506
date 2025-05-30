"""Run a Python code snippet (in a file) on an ARC AGI problem"""

import argparse
import copy
import pickle

import joblib
import numpy as np

import utils
from utils import ExecutionOutcome, RunResult


def exec_and_run(code_as_line, initial):
    code_as_line = "import numpy as np\n" + code_as_line
    # import numpy as np
    # if numpy not in globals (outside of this fn) then
    # this fails, if I add it to the code then it works
    # if I add locals then I can't inject the new code
    exec(code_as_line, globals())

    # result = None
    result = transform(initial)
    # try:
    #    result = transform(initial)
    # except (NameError, AssertionError):
    #    # AssertionError almost certainly on the size of the array checks
    #    pass
    return result


def execute_transform(code_as_line, problems):
    """Run the code on training problems (with train/test pair), output a summary"""
    transform_at_least_one_ran = False  # we succeeded on at least one input
    transform_all_ran_and_matched = (
        False  # 1 if we succeeded on all inputs to match desired final
    )
    partial_score = 0  # how often did transform correctly turn initial into final?
    code_ran_on_all_inputs = False  # this code ran on all inputs, 1 if true
    code_executed = False  # did code_as_line run through exec without error?
    # transform = None  # dummy definition to keep syntax highligher happy
    execution_outcomes = []  # build this up one item at a time
    exception_message = None  # None if no exception raised

    try:
        exec(code_as_line, globals())  # try to generate transform function
        # transform # check if transform is in namespace
        # print(f"transform in namespace: {'transform' in dir()}")
        # if i exec to locals, dir() sees it but i can't call it
        # if i exec to globals, dir() doesn't see it but i can call it
        code_executed = True  # if we get here, the code could be executed

        n_jobs = max(len(problems), 2)

        try:
            # n_jobs = 1 for serial process
            # result_chunks = joblib.Parallel(n_jobs=len(problems), timeout=1)(
            result_chunks = joblib.Parallel(n_jobs=n_jobs, timeout=1)(
                joblib.delayed(exec_and_run)(
                    code_as_line, np.array(copy.deepcopy(prob["input"]))
                )
                for prob in problems
            )
        except (RuntimeError, pickle.PicklingError, joblib.parallel.TimeoutError) as e:
            # joblib was unhappy, possibly because of a raw_input
            # e.g. RuntimeError: input(): lost sys.stdin
            # also loxy error PicklingError: Could not pickle the task to send it to the workers.
            # TimeoutError occurs for
            # "exp_2024-07-02T08_Meta-Llama-3-70B-Instruct-IQ1_S.gguf_a85_listgrid"
            # due to infinite loop
            pass

        for prob_n, prob in enumerate(problems):
            # wrapped in copies as a self protection mechanism
            # initial = copy.deepcopy(prob["input"]) # initial
            # final_gen = transform(initial)  # the generated final output
            final_gen = result_chunks[prob_n]
            if not isinstance(final_gen, np.ndarray):
                raise ValueError(
                    f"Expecting `transform` to return a numpy array, got {type(final)}."
                )
            # final = np.array(copy.deepcopy(prob["output"]))  # desired final
            final = np.array(copy.deepcopy(prob["output"]))  # desired final

            was_correct = (final_gen == final).all()
            if was_correct:
                partial_score += 1  # we get a point if we match the desired final
            initial = prob["input"]
            eo = ExecutionOutcome(initial, final, final_gen, was_correct)
            # breakpoint()
            execution_outcomes.append(eo)
        code_ran_on_all_inputs = True  # note that we correctly ran on all inputs
    except (
        SyntaxError,
        AttributeError,
        TypeError,
        KeyError,
        NameError,
        ValueError,
        IndexError,
        NotImplementedError,
        OverflowError,
        AssertionError,
        ZeroDivisionError,
        ModuleNotFoundError,
        Exception,  # llama actually wrote code to raise an Exception!
    ) as e:
        exception_message = f"execute_transform caught: {e} of {type(e)=}"
        print(exception_message)
        # print(e, type(e))
        # pprint.pprint(code_as_line)

    if partial_score > 0:
        transform_at_least_one_ran = True

    if partial_score == len(problems):
        transform_all_ran_and_matched = True

    rr = RunResult(
        code_executed,
        code_ran_on_all_inputs,
        transform_all_ran_and_matched,
        transform_at_least_one_ran,
        partial_score,
    )
    return rr, execution_outcomes, exception_message


#    return code_executed, code_ran_on_all_inputs, transform_all_ran_and_matched, transform_at_least_one_ran, partial_score


if __name__ == "__main__":

    parser = utils.add_argument_parser(problem_name=True, code_filename=True)

    args = parser.parse_args()

    # Load code from the specified file
    try:
        with open(args.code_filename, "r") as f:
            code_as_string = f.read()
            print(
                f"Loaded {len(code_as_string.split('\n'))} lines of code from {args.code_filename}"
            )
    except FileNotFoundError:
        print(f"Error: The file {args.code_filename} does not exist.")
        print(
            "This means you haven't setup an input file of code to execute (i.e. a transform function in a file)"
        )
        exit(1)

    problem_train_test = utils.get_examples(args.problem_name)
    train_problems = problem_train_test["train"]
    test_problems = problem_train_test["test"]

    print(f"Executing {args.problem_name} with {args.code_filename}")
    print(f"Code:\n{code_as_string}")
    print(f"Running on train problems: {len(train_problems)}")
    rr_train = execute_transform(code_as_string, train_problems)
    print(rr_train)
    # print(f"Running on test problems: {len(test_problems)}")
    # rr_test = execute_transform(code_as_string, test_problems)
    # print(rr_test)
