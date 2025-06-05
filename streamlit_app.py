import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import traceback
from pathlib import Path

# Import existing modules
import utils
from run_code import execute_transform
from prompt import make_prompt, get_func_dict
from method1_text_prompt import run_experiment_for_iterations
from config import providers
from litellm_helper import check_litellm_key
import argparse

# Configure page
st.set_page_config(
    page_title="ARC Challenge Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better grid visualization
st.markdown(
    """
<style>
.grid-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 10px;
}
.grid-title {
    font-weight: bold;
    margin-bottom: 5px;
    text-align: center;
}
.status-correct {
    color: green;
    font-weight: bold;
}
.status-incorrect {
    color: red;
    font-weight: bold;
}
.code-editor {
    font-family: 'Courier New', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)

# Color mapping for ARC grids (0-9)
ARC_COLORS = {
    0: "#000000",  # Black
    1: "#0074D9",  # Blue
    2: "#FF4136",  # Red
    3: "#2ECC40",  # Green
    4: "#FFDC00",  # Yellow
    5: "#AAAAAA",  # Gray
    6: "#F012BE",  # Fuchsia
    7: "#FF851B",  # Orange
    8: "#7FDBFF",  # Aqua
    9: "#870C25",  # Maroon
}


def plot_grid(grid, title="Grid", figsize=(4, 4)):
    """Plot an ARC grid with proper colors"""
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map
    colors = [ARC_COLORS[i] for i in range(10)]
    cmap = mcolors.ListedColormap(colors)

    # Plot the grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add numbers in cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                str(grid[i, j]),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )

    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def get_available_problems():
    """Get list of available ARC problems"""
    try:
        path = "./arc_data/arc-prize-2025/arc-agi_training_challenges.json"
        with open(path) as f:
            data = json.load(f)
        problems = list(data.keys())
        return problems
    except Exception as e:
        st.error(f"Error loading problems: {e}")
        print(f"ERROR: Error in get_available_problems: {e}")
        return []


def load_problem_data(problem_id):
    """Load a specific problem's data"""
    try:
        result = utils.get_examples(problem_id)
        return result
    except KeyError as e:
        error_msg = f"Problem ID '{problem_id}' not found in dataset. Available problems can be seen in the dropdown."
        print(f"ERROR: KeyError for problem {problem_id}: {e}")
        st.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error loading problem {problem_id}: {e}"
        print(f"ERROR: Exception loading problem {problem_id}: {e}")
        st.error(error_msg)
        return None


def display_training_examples(problem_data):
    """Display training examples for a problem"""
    st.subheader("Training Examples")

    train_problems = problem_data["train"]

    for i, example in enumerate(train_problems):
        st.write(f"**Example {i+1}:**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Input:")
            fig_input = plot_grid(example["input"], f"Input {i+1}")
            st.pyplot(fig_input)
            plt.close()

        with col2:
            st.write("Output:")
            fig_output = plot_grid(example["output"], f"Output {i+1}")
            st.pyplot(fig_output)
            plt.close()

        st.write("---")


def display_test_examples(problem_data):
    """Display test examples for a problem"""
    st.subheader("Test Examples")

    test_problems = problem_data["test"]

    for i, example in enumerate(test_problems):
        st.write(f"**Test {i+1}:**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Input:")
            fig_input = plot_grid(example["input"], f"Test Input {i+1}")
            st.pyplot(fig_input)
            plt.close()

        with col2:
            if "output" in example:
                st.write("Expected Output:")
                fig_output = plot_grid(example["output"], f"Expected Output {i+1}")
                st.pyplot(fig_output)
                plt.close()
            else:
                st.write("Output: *Hidden*")


def display_execution_results(rr, execution_outcomes, exception_message):
    """Display the results of code execution"""
    st.subheader("Execution Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Code Executed", "✅" if rr.code_did_execute else "❌")

    with col2:
        st.metric("Ran on All Inputs", "✅" if rr.code_ran_on_all_inputs else "❌")

    with col3:
        st.metric(
            "All Correct", "✅" if rr.transform_ran_and_matched_for_all_inputs else "❌"
        )

    with col4:
        st.metric(
            "Score",
            f"{rr.transform_ran_and_matched_score}/{len(execution_outcomes) if execution_outcomes else 0}",
        )

    # Show exception if any
    if exception_message:
        st.error(f"Error: {exception_message}")

    # Show individual results
    if execution_outcomes:
        st.write("**Individual Results:**")

        for i, eo in enumerate(execution_outcomes):
            st.write(f"**Example {i+1}:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Input:")
                fig_input = plot_grid(eo.initial, f"Input {i+1}", figsize=(3, 3))
                st.pyplot(fig_input)
                plt.close()

            with col2:
                st.write("Expected:")
                fig_expected = plot_grid(eo.final, f"Expected {i+1}", figsize=(3, 3))
                st.pyplot(fig_expected)
                plt.close()

            with col3:
                st.write("Generated:")
                if eo.generated_final is not None:
                    fig_generated = plot_grid(
                        eo.generated_final, f"Generated {i+1}", figsize=(3, 3)
                    )
                    st.pyplot(fig_generated)
                    plt.close()

                    # Show status
                    if eo.was_correct:
                        st.markdown(
                            '<p class="status-correct">✅ Correct</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<p class="status-incorrect">❌ Incorrect</p>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("*No output generated*")

            st.write("---")


def get_default_code():
    """Get default transform function code"""
    return """import numpy as np

def transform(initial):
    assert isinstance(initial, np.ndarray)
    initial = np.array(initial)

    # Your solution code here
    final = initial.copy()  # placeholder - replace with your logic

    assert isinstance(final, np.ndarray)
    return final"""


def main():
    st.title("🧩 ARC Challenge Solver")
    st.write(
        "Interactive visualization and solving environment for ARC (Abstraction and Reasoning Corpus) challenges"
    )

    # Sidebar for problem selection
    st.sidebar.header("Problem Selection")

    # Load available problems
    available_problems = get_available_problems()

    if not available_problems:
        st.error("No problems found. Please check your data directory.")
        st.error(
            "Expected path: ./arc_data/arc-prize-2025/arc-agi_training_challenges.json"
        )
        return

    # Add some common problems at the top for easy access
    common_problems = [
        "9565186b",
        "0d3d703e",
        "08ed6ac7",
        "0a938d79",
        "178fcbfb",
        "1a07d186",
    ]
    common_available = [p for p in common_problems if p in available_problems]
    other_problems = [p for p in available_problems if p not in common_problems]

    problem_options = (
        ["Select a problem..."] + common_available + ["---"] + other_problems
    )

    selected_problem = st.sidebar.selectbox(
        "Choose an ARC problem:", problem_options, key="problem_selector"
    )

    if selected_problem == "Select a problem..." or selected_problem == "---":
        st.info("👈 Please select a problem from the sidebar to begin")
        return

    # Load problem data
    problem_data = load_problem_data(selected_problem)
    if problem_data is None:
        st.error(f"Failed to load problem data for: {selected_problem}")
        return

    st.header(f"Problem: {selected_problem}")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Problem View", "💻 Code Editor", "🤖 LLM Assistant", "📊 Results"]
    )

    with tab1:
        # Display training and test examples
        display_training_examples(problem_data)
        display_test_examples(problem_data)

    with tab2:
        st.subheader("Code Editor")
        st.write("Write your `transform` function to solve the problem:")

        # Add example code loader
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button(
                "📄 Load Example Code",
                help="Load the working code from code_3_example.py",
            ):
                try:
                    with open("code_3_example.py", "r") as f:
                        example_code = f.read()
                    st.session_state.user_code = example_code
                    st.success("Loaded example code from code_3_example.py!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading example code: {e}")

        with col_b:
            if st.button("🔄 Reset to Default", help="Reset to the default template"):
                st.session_state.user_code = get_default_code()
                st.success("Reset to default template!")
                st.rerun()

        # Code editor
        if "user_code" not in st.session_state:
            st.session_state.user_code = get_default_code()

        user_code = st.text_area(
            "Transform Function:",
            value=st.session_state.user_code,
            height=300,
            help="Write a function that transforms the input grid to the output grid",
        )
        st.session_state.user_code = user_code

        # Execution controls
        col1, col2 = st.columns([1, 3])

        with col1:
            execute_button = st.button("🚀 Execute Code", type="primary")

        with col2:
            st.write("*This will run your code on all training examples*")

        if execute_button:
            if user_code.strip():
                with st.spinner("Executing code on training examples..."):
                    try:
                        # Execute the transform function
                        train_problems = problem_data["train"]
                        rr, execution_outcomes, exception_message = execute_transform(
                            user_code, train_problems
                        )

                        # Store results in session state
                        st.session_state.execution_results = (
                            rr,
                            execution_outcomes,
                            exception_message,
                        )
                        st.session_state.has_results = True

                        st.success("Code execution completed!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Execution failed: {str(e)}")
                        st.write("**Traceback:**")
                        st.code(traceback.format_exc())
            else:
                st.warning("Please enter some code first!")

        # Display results if available
        if st.session_state.get("has_results", False):
            rr, execution_outcomes, exception_message = (
                st.session_state.execution_results
            )
            display_execution_results(rr, execution_outcomes, exception_message)

            # Add validation section for code_3_example.py on problem 9565186b
            if selected_problem == "9565186b" and rr.code_ran_on_all_inputs:
                st.subheader("🔍 Validation Check")

                expected_score = 4
                expected_all_correct = True

                if (
                    rr.transform_ran_and_matched_score == expected_score
                    and rr.transform_ran_and_matched_for_all_inputs
                    == expected_all_correct
                ):
                    st.success(
                        f"✅ VALIDATION PASSED: Results match command line execution!"
                    )
                    st.success(
                        f"Score: {rr.transform_ran_and_matched_score}/{len(execution_outcomes)} (Expected: {expected_score}/4)"
                    )
                else:
                    st.error(f"❌ VALIDATION FAILED: Results don't match expected")
                    st.error(
                        f"Score: {rr.transform_ran_and_matched_score}/{len(execution_outcomes)} (Expected: {expected_score}/4)"
                    )
                    st.error(
                        f"All correct: {rr.transform_ran_and_matched_for_all_inputs} (Expected: {expected_all_correct})"
                    )

                # Show detailed comparison for first example
                if execution_outcomes and len(execution_outcomes) > 0:
                    eo = execution_outcomes[0]
                    expected_input = [[2, 2, 2], [8, 8, 2], [2, 2, 2]]
                    expected_output = [[2, 2, 2], [5, 5, 2], [2, 2, 2]]

                    st.write("**First Example Validation:**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("Input matches:")
                        input_matches = np.array_equal(eo.initial, expected_input)
                        st.write("✅" if input_matches else "❌", input_matches)

                    with col2:
                        st.write("Expected output matches:")
                        expected_matches = np.array_equal(eo.final, expected_output)
                        st.write("✅" if expected_matches else "❌", expected_matches)

                    with col3:
                        st.write("Generated output matches:")
                        if eo.generated_final is not None:
                            generated_matches = np.array_equal(
                                eo.generated_final, expected_output
                            )
                            st.write(
                                "✅" if generated_matches else "❌", generated_matches
                            )
                        else:
                            st.write("❌ No output generated")

    with tab3:
        st.subheader("🤖 LLM Assistant")
        st.write("Use AI to help solve the problem")

        # LLM Configuration
        st.write("**Configuration:**")
        col1, col2 = st.columns(2)

        with col1:
            model_options = list(providers.keys())
            selected_model = st.selectbox("Model:", model_options, index=0)

        with col2:
            iterations = st.number_input(
                "Iterations:", min_value=1, max_value=10, value=1
            )

        # Template selection
        template_files = [f.name for f in Path("templates").glob("*.txt")]
        selected_template = st.selectbox("Prompt Template:", template_files, index=0)

        # Generate solution button
        if st.button("🎯 Generate Solution", type="primary"):
            with st.spinner(
                f"Running {iterations} iteration(s) with {selected_model}..."
            ):
                try:
                    # Create a mock args object for the existing functions
                    class MockArgs:
                        def __init__(self):
                            self.model_name = selected_model

                    mock_args = MockArgs()

                    # Run the LLM experiment
                    llm_responses, rr_trains = run_experiment_for_iterations(
                        model=selected_model,
                        provider=providers[selected_model],
                        iterations=iterations,
                        problems=problem_data,
                        template_name=selected_template,
                    )

                    # Display results
                    st.success(f"Generated {len(llm_responses)} solution(s)")

                    for i, (response, rr_train) in enumerate(
                        zip(llm_responses, rr_trains)
                    ):
                        st.write(f"**Attempt {i+1}:**")

                        # Show LLM response
                        with st.expander(f"LLM Response {i+1}"):
                            st.write(response.choices[0].message.content)

                        # Show execution results
                        rr, execution_outcomes, exception_message = rr_train

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Code Executed", "✅" if rr.code_did_execute else "❌"
                            )
                        with col2:
                            st.metric(
                                "All Correct",
                                (
                                    "✅"
                                    if rr.transform_ran_and_matched_for_all_inputs
                                    else "❌"
                                ),
                            )
                        with col3:
                            st.metric(
                                "Score",
                                f"{rr.transform_ran_and_matched_score}/{len(execution_outcomes) if execution_outcomes else 0}",
                            )

                        # If successful, offer to copy code
                        if rr.transform_ran_and_matched_for_all_inputs:
                            st.success("🎉 This solution works!")
                            if st.button(
                                f"Copy Solution {i+1} to Editor", key=f"copy_{i}"
                            ):
                                # Extract code from response
                                code = utils.extract_from_code_block(
                                    response.choices[0].message.content
                                )
                                if code:
                                    st.session_state.user_code = code
                                    st.success("Code copied to editor!")
                                    st.rerun()

                        st.write("---")

                except Exception as e:
                    st.error(f"LLM generation failed: {str(e)}")
                    st.write("**Traceback:**")
                    st.code(traceback.format_exc())

    with tab4:
        st.subheader("📊 Analysis & Statistics")

        if st.session_state.get("has_results", False):
            rr, execution_outcomes, exception_message = (
                st.session_state.execution_results
            )

            # Detailed analysis
            st.write("**Detailed Analysis:**")

            if execution_outcomes:
                # Success rate
                correct_count = sum(1 for eo in execution_outcomes if eo.was_correct)
                total_count = len(execution_outcomes)
                success_rate = correct_count / total_count if total_count > 0 else 0

                st.metric(
                    "Success Rate",
                    f"{success_rate:.1%} ({correct_count}/{total_count})",
                )

                # Grid size analysis
                st.write("**Grid Size Analysis:**")
                for i, eo in enumerate(execution_outcomes):
                    input_shape = eo.initial.shape
                    expected_shape = eo.final.shape
                    generated_shape = (
                        eo.generated_final.shape
                        if eo.generated_final is not None
                        else "None"
                    )

                    st.write(
                        f"Example {i+1}: Input {input_shape} → Expected {expected_shape}, Generated {generated_shape}"
                    )

                # Pattern analysis
                st.write("**Pattern Analysis:**")

                # Check for size transformations
                size_changes = []
                for eo in execution_outcomes:
                    input_size = np.prod(eo.initial.shape)
                    output_size = np.prod(eo.final.shape)
                    size_changes.append(output_size / input_size)

                if len(set(size_changes)) == 1:
                    ratio = size_changes[0]
                    if ratio == 1:
                        st.info("🔍 Pattern: Same size transformation")
                    elif ratio > 1:
                        st.info(f"🔍 Pattern: Size increase by factor of {ratio:.1f}")
                    else:
                        st.info(f"🔍 Pattern: Size decrease by factor of {1/ratio:.1f}")
                else:
                    st.info("🔍 Pattern: Variable size transformation")
        else:
            st.info("Execute some code first to see analysis results")


if __name__ == "__main__":
    # Initialize session state
    if "has_results" not in st.session_state:
        st.session_state.has_results = False

    main()
