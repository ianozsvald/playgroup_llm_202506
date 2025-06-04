import utils
from prompt_feedback import CODE_9565186b, CODE_9565186b_bad, feedback_on_executions
from run_code import execute_transform


def test_feedback_on_good_code():
    patterns = utils.get_examples("9565186b")

    problems = patterns["train"]

    rr, execution_outcomes, exception_message = execute_transform(
        CODE_9565186b, problems
    )
    assert rr.code_did_execute is True

    rr_train = (rr, execution_outcomes, exception_message)
    output = feedback_on_executions(rr_train)
    assert "You got this example CORRECT" in output
    assert "You had 4 examples that executed" in output
    print(output)


def test_feedback_on_bad_code():
    patterns = utils.get_examples("9565186b")

    problems = patterns["train"]
    rr, execution_outcomes, exception_message = execute_transform(
        CODE_9565186b_bad, problems
    )
    # Out[8]: run_result(code_did_execute=False, code_ran_on_all_inputs=False,
    # transform_ran_and_matched_for_all_inputs=False, transform_ran_and_matched_at_least_once=False,
    # transform_ran_and_matched_score=0)
    assert rr.code_did_execute is False
    rr_train = (rr, execution_outcomes, exception_message)
    output = feedback_on_executions(rr_train)
    assert (
        "You had no examples that executed, your code is bad and must be fixed"
        in output
    )
    assert "No module named 'scipy'" in exception_message
    print(output)
