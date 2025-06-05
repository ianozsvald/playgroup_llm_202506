import streamlit as st
import numpy as np
import pandas as pd
import json
import traceback
from pathlib import Path

# Import existing modules
import utils
from run_code import execute_transform
from prompt import make_prompt, get_func_dict
from method1_text_prompt import run_experiment_for_iterations
from config import providers
from litellm_helper import check_litellm_key, call_llm
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


def display_grid_dataframe(grid, title="Grid", compact=False):
    """Display grid as a colored dataframe"""
    if not grid:
        st.write("Empty grid")
        return

    grid = np.array(grid)
    df = pd.DataFrame(grid)

    # Create color styling function
    def color_cells(val):
        color = ARC_COLORS.get(val, "#FFFFFF")
        # Use white text for dark colors, black text for light colors
        # Dark colors: 0 (black), 2 (red), 9 (maroon), 1 (blue)
        text_color = "white" if val in [0, 1, 2, 9] else "black"
        font_size = "12px" if compact else "16px"
        padding = "4px 8px" if compact else "8px 12px"
        return f"background-color: {color}; color: {text_color}; font-weight: bold; text-align: center; font-size: {font_size}; padding: {padding}; border: 1px solid #333;"

    if title:
        st.write(f"**{title}**")

    # Display the styled dataframe
    styled_df = df.style.applymap(color_cells)
    styled_df = styled_df.set_table_styles(
        [
            {"selector": "th", "props": [("display", "none")]},  # Hide column headers
            {"selector": "td", "props": [("border", "1px solid #333")]},  # Add borders
            {
                "selector": "",
                "props": [("border-collapse", "collapse")],
            },  # Collapse borders
        ]
    )

    st.dataframe(styled_df, use_container_width=False, hide_index=True)


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
            display_grid_dataframe(example["input"], f"Input {i+1}")

        with col2:
            st.write("Output:")
            display_grid_dataframe(example["output"], f"Output {i+1}")

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
            display_grid_dataframe(example["input"], f"Test Input {i+1}")

        with col2:
            if "output" in example:
                st.write("Expected Output:")
                display_grid_dataframe(example["output"], f"Expected Output {i+1}")
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
                display_grid_dataframe(eo.initial, f"Input {i+1}", compact=True)

            with col2:
                st.write("Expected:")
                display_grid_dataframe(eo.final, f"Expected {i+1}", compact=True)

            with col3:
                st.write("Generated:")
                if eo.generated_final is not None:
                    display_grid_dataframe(
                        eo.generated_final, f"Generated {i+1}", compact=True
                    )

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Problem View",
            "💻 Code Editor",
            "🤖 LLM Assistant",
            "✍️ Prompt Lab",
            "📊 Results",
        ]
    )

    with tab1:
        # Display training and test examples
        display_training_examples(problem_data)
        display_test_examples(problem_data)

    with tab2:
        st.subheader("Code Editor")
        st.write("Write your `transform` function to solve the problem:")

        # Add example code loader
        col_a, col_b, col_c = st.columns([1, 1, 1])
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
            # Load saved generated code
            saved_code_files = []
            if Path("saved_code").exists():
                saved_code_files = [f.name for f in Path("saved_code").glob("*.py")]

            if saved_code_files:
                selected_saved_code = st.selectbox(
                    "Load saved code:",
                    ["Select..."] + saved_code_files,
                    key="saved_code_selector",
                    help="Load previously saved generated code",
                )

                if st.button("📂 Load Saved", help="Load the selected saved code"):
                    if selected_saved_code != "Select...":
                        try:
                            with open(f"saved_code/{selected_saved_code}", "r") as f:
                                saved_code_content = f.read()

                            # Extract just the code part (remove metadata comments)
                            lines = saved_code_content.split("\n")
                            code_start = 0
                            for i, line in enumerate(lines):
                                if line.strip() and not line.strip().startswith("#"):
                                    code_start = i
                                    break

                            clean_code = "\n".join(lines[code_start:])
                            st.session_state.user_code = clean_code
                            st.success(f"Loaded saved code: {selected_saved_code}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading saved code: {e}")
                    else:
                        st.warning("Please select a saved code file")
            else:
                st.info("No saved code found")

        with col_c:
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

                        # Show full LLM response prominently
                        st.write("### 🤖 Complete LLM Response")
                        full_response = response.choices[0].message.content
                        with st.expander("Full LLM Response", expanded=True):
                            st.markdown(full_response)

                        # Extract and test code
                        code_as_string = utils.extract_from_code_block(full_response)

                        if code_as_string:
                            st.write("### 💻 Extracted Code")
                            with st.expander("Generated Code", expanded=True):
                                st.code(code_as_string, language="python")

                            st.info("🔧 Testing code execution...")

                            # Execute code
                            train_problems = problem_data["train"]
                            rr, execution_outcomes, exception_message = (
                                execute_transform(code_as_string, train_problems)
                            )

                            # Show quick results
                            st.write("### 📊 Execution Results")
                            col_result1, col_result2, col_result3 = st.columns(3)
                            with col_result1:
                                st.metric(
                                    "Code Executed",
                                    "✅" if rr.code_did_execute else "❌",
                                )
                            with col_result2:
                                st.metric(
                                    "All Correct",
                                    (
                                        "✅"
                                        if rr.transform_ran_and_matched_for_all_inputs
                                        else "❌"
                                    ),
                                )
                            with col_result3:
                                st.metric(
                                    "Score",
                                    f"{rr.transform_ran_and_matched_score}/{len(execution_outcomes) if execution_outcomes else 0}",
                                )

                            # Store results for detailed view
                            st.session_state.prompt_test_results = (
                                rr,
                                execution_outcomes,
                                exception_message,
                                response,
                                code_as_string,
                            )

                            # Add option to save the generated code
                            if rr.transform_ran_and_matched_score > 0:  # Any success
                                st.write("### 💾 Save Generated Code")
                                col_save_code, col_save_btn = st.columns([2, 1])

                                with col_save_code:
                                    save_code_name = st.text_input(
                                        "Code name:",
                                        value=f"generated_{selected_problem}_{st.session_state.custom_prompt_name}",
                                        key="save_code_name_input",
                                        help="Name for saving the generated code",
                                    )

                                with col_save_btn:
                                    if st.button(
                                        "💾 Save Code", key="save_generated_code"
                                    ):
                                        if save_code_name.strip():
                                            try:
                                                # Create saved_code directory if it doesn't exist
                                                import os

                                                os.makedirs("saved_code", exist_ok=True)

                                                # Save with metadata
                                                filename = f"saved_code/{save_code_name.strip()}.py"
                                                metadata = f"""# Generated code saved from Prompt Lab
# Problem: {selected_problem}
# Template: {st.session_state.custom_prompt_name}
# Score: {rr.transform_ran_and_matched_score}/{len(execution_outcomes)}
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
                                                with open(filename, "w") as f:
                                                    f.write(metadata + code_as_string)
                                                st.success(
                                                    f"✅ Code saved as: {filename}"
                                                )
                                            except Exception as e:
                                                st.error(f"❌ Save failed: {e}")
                                        else:
                                            st.warning("Please enter a code name")

                            if exception_message:
                                st.error(f"Execution error: {exception_message}")
                        else:
                            st.warning("⚠️ No code block found in LLM response")
                            st.write("**Raw Response Preview:**")
                            st.text(
                                full_response[:500] + "..."
                                if len(full_response) > 500
                                else full_response
                            )
                    else:
                        st.error("❌ No response from LLM")
                except Exception as e:
                    st.error(f"LLM generation failed: {str(e)}")
                    st.write("**Traceback:**")
                    st.code(traceback.format_exc())

    with tab4:
        st.subheader("✍️ Prompt Lab")
        st.write("Create, edit, test, and save custom prompts for LLM code generation")

        # Show current problem for reference
        if problem_data:
            st.write("### 🧩 Current Problem Reference")
            with st.expander(
                f"Problem {selected_problem} - Training Examples", expanded=False
            ):
                train_problems = problem_data["train"]
                for i, example in enumerate(train_problems):
                    st.write(f"**Example {i+1}:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Input:")
                        display_grid_dataframe(
                            example["input"], f"Input {i+1}", compact=True
                        )

                    with col2:
                        st.write("Output:")
                        display_grid_dataframe(
                            example["output"], f"Output {i+1}", compact=True
                        )

                    if (
                        i < len(train_problems) - 1
                    ):  # Don't add separator after last example
                        st.write("---")
        else:
            st.info(
                "👈 Select a problem from the sidebar to see training examples here"
            )

        # Template management section
        st.write("### Template Management")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Template selection
            template_files = [f.name for f in Path("templates").glob("*.txt")]
            if template_files:
                selected_template_lab = st.selectbox(
                    "Load existing template:",
                    ["Create new..."] + template_files,
                    key="template_lab_selector",
                )
            else:
                selected_template_lab = "Create new..."
                st.info("No existing templates found")

        with col2:
            if st.button(
                "📁 Load Template", help="Load the selected template for editing"
            ):
                if selected_template_lab != "Create new...":
                    try:
                        template_path = f"templates/{selected_template_lab}"
                        with open(template_path, "r") as f:
                            template_content = f.read()

                        # Force update session state
                        st.session_state.custom_prompt = template_content
                        st.session_state.custom_prompt_name = (
                            selected_template_lab.replace(".txt", "")
                        )

                        # Clear any cached data
                        if "preview_prompt" in st.session_state:
                            del st.session_state.preview_prompt
                        if "prompt_test_results" in st.session_state:
                            del st.session_state.prompt_test_results

                        st.success(f"✅ Loaded template: {selected_template_lab}")
                        st.info("📝 Template content updated in editor below")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading template: {e}")
                        st.error(f"Path attempted: {template_path}")
                else:
                    st.info("Select a template file to load")

        with col3:
            # Force refresh button for debugging
            if st.button(
                "🔄 Force Refresh", help="Clear cache and refresh template list"
            ):
                # Clear relevant session state
                if "preview_prompt" in st.session_state:
                    del st.session_state.preview_prompt
                if "prompt_test_results" in st.session_state:
                    del st.session_state.prompt_test_results
                st.success("🔄 Cache cleared!")
                st.rerun()

        # Template name input
        if "custom_prompt_name" not in st.session_state:
            st.session_state.custom_prompt_name = "my_custom_prompt"

        new_name = st.text_input(
            "Template name:",
            value=st.session_state.custom_prompt_name,
            key="prompt_name_input",
            help="Name for saving the template (without .txt extension)",
        )
        st.session_state.custom_prompt_name = new_name

        # Template editor
        st.write("### Template Editor")

        # Initialize custom prompt if not exists
        if "custom_prompt" not in st.session_state:
            # Default template
            st.session_state.custom_prompt = """You are a clever problem solving machine. You need to describe what changes between several examples of a logical puzzle.

Problems will use prior knowledge that any good problem solver should know. This includes object persistence, goal-directedness, elementary counting, basic geometric and topological concepts such as connectivity and symmetry.

You'll see some input and output pairs for a grid of numbers.

{% set grid_method = make_grid_plain -%}
{% for pattern_input_output in patterns_input_output %}
Here is an example input and output pattern as a JSON dict:
{{ pattern_input_output }}
and then as the input grid:
{{ grid_method(pattern_input_output['input']) }}
and a corresponding output grid:
{{ grid_method(pattern_input_output['output']) }}
{% endfor -%}

Given the above examples, write several bullet points that explain the rules that convert the input patterns to the output patterns.

After this write a solution in Python code that follows the following format. You must accept an `initial` np.ndarray of numbers as input and return a `final` np.ndarray of numbers. Each number is in the range [0...9] and the grid is rectangular.

```python
import numpy as np
def transform(initial):
    assert isinstance(initial, np.ndarray)
    ... # you need to write code to generate `final`
    assert isinstance(final, np.ndarray)
    return final
```"""

        # Template editor with syntax help
        col_editor, col_help = st.columns([2, 1])

        with col_editor:
            custom_prompt = st.text_area(
                "Prompt Template (Jinja2 format):",
                value=st.session_state.custom_prompt,
                height=400,
                help="Use Jinja2 syntax. Available variables: patterns_input_output, grid formatting functions",
            )
            st.session_state.custom_prompt = custom_prompt

        with col_help:
            st.write("**Available Variables:**")
            st.code("{{ patterns_input_output }}", language="jinja2")
            st.write("List of input/output pattern dicts")

            st.write("**Grid Formatting Functions:**")
            st.code("{{ make_grid_plain(grid) }}", language="jinja2")
            st.write("Simple grid: 123\\n456")

            st.code("{{ make_grid_csv(grid) }}", language="jinja2")
            st.write("CSV format: 1, 2, 3\\n4, 5, 6")

            st.code("{{ make_grid_csv_quoted(grid) }}", language="jinja2")
            st.write('Quoted CSV: "1", "2", "3"')

            st.code("{{ make_grid_csv_english_words(grid) }}", language="jinja2")
            st.write("Words: one, two, three")

            st.write("**Jinja2 Syntax:**")
            st.code(
                """{% for item in list %}
{{ item }}
{% endfor %}""",
                language="jinja2",
            )

            st.code("{% set var = value %}", language="jinja2")

        # Template preview and testing
        st.write("### Template Preview & Testing")

        col_preview, col_test = st.columns([1, 1])

        with col_preview:
            if st.button(
                "🔍 Preview Prompt",
                help="Preview how the prompt will look with current problem data",
            ):
                if problem_data:
                    try:
                        from jinja2 import Environment, BaseLoader, Template

                        # Create template
                        template = Template(custom_prompt)
                        func_dict = get_func_dict()
                        template.globals.update(func_dict)

                        # Render with current problem data
                        rendered_prompt = template.render(
                            patterns_input_output=problem_data["train"]
                        )

                        st.success("✅ Template rendered successfully!")

                        with st.expander("Preview Rendered Prompt", expanded=True):
                            st.text_area(
                                "Rendered Prompt:",
                                value=rendered_prompt,
                                height=300,
                                disabled=True,
                            )

                        # Store for testing
                        st.session_state.preview_prompt = rendered_prompt

                    except Exception as e:
                        st.error(f"❌ Template rendering failed: {e}")
                        st.code(str(e))
                else:
                    st.warning("Please select a problem first to preview the template")

        with col_test:
            if st.button(
                "🧪 Test Prompt",
                help="Test the prompt with LLM and execute generated code",
            ):
                if problem_data and st.session_state.get("preview_prompt"):
                    with st.spinner("Testing prompt with LLM..."):
                        try:
                            # Get LLM settings from main tab
                            model_options = list(providers.keys())
                            test_model = model_options[0] if model_options else None

                            if test_model:
                                # Create messages for LLM
                                content = [
                                    {
                                        "type": "text",
                                        "text": st.session_state.preview_prompt,
                                    }
                                ]
                                messages = [{"content": content, "role": "user"}]

                                # Call LLM
                                response = call_llm(
                                    test_model, messages, providers[test_model]
                                )

                                if response:
                                    st.success("✅ LLM responded successfully!")

                                    # Show full response prominently
                                    st.write("### 🤖 Complete LLM Response")
                                    full_response = response.choices[0].message.content
                                    with st.expander(
                                        "Full LLM Response", expanded=True
                                    ):
                                        st.markdown(full_response)

                                    # Extract and test code
                                    code_as_string = utils.extract_from_code_block(
                                        full_response
                                    )

                                    if code_as_string:
                                        st.write("### 💻 Extracted Code")
                                        with st.expander(
                                            "Generated Code", expanded=True
                                        ):
                                            st.code(code_as_string, language="python")

                                        st.info("🔧 Testing code execution...")

                                        # Execute code
                                        train_problems = problem_data["train"]
                                        rr, execution_outcomes, exception_message = (
                                            execute_transform(
                                                code_as_string, train_problems
                                            )
                                        )

                                        # Show quick results
                                        st.write("### 📊 Execution Results")
                                        col_result1, col_result2, col_result3 = (
                                            st.columns(3)
                                        )
                                        with col_result1:
                                            st.metric(
                                                "Code Executed",
                                                "✅" if rr.code_did_execute else "❌",
                                            )
                                        with col_result2:
                                            st.metric(
                                                "All Correct",
                                                (
                                                    "✅"
                                                    if rr.transform_ran_and_matched_for_all_inputs
                                                    else "❌"
                                                ),
                                            )
                                        with col_result3:
                                            st.metric(
                                                "Score",
                                                f"{rr.transform_ran_and_matched_score}/{len(execution_outcomes) if execution_outcomes else 0}",
                                            )

                                        # Store results for detailed view
                                        st.session_state.prompt_test_results = (
                                            rr,
                                            execution_outcomes,
                                            exception_message,
                                            response,
                                            code_as_string,
                                        )

                                        # Add option to save the generated code
                                        if (
                                            rr.transform_ran_and_matched_score > 0
                                        ):  # Any success
                                            st.write("### 💾 Save Generated Code")
                                            col_save_code, col_save_btn = st.columns(
                                                [2, 1]
                                            )

                                            with col_save_code:
                                                save_code_name = st.text_input(
                                                    "Code name:",
                                                    value=f"generated_{selected_problem}_{st.session_state.custom_prompt_name}",
                                                    key="save_code_name_input",
                                                    help="Name for saving the generated code",
                                                )

                                            with col_save_btn:
                                                if st.button(
                                                    "💾 Save Code",
                                                    key="save_generated_code",
                                                ):
                                                    if save_code_name.strip():
                                                        try:
                                                            # Create saved_code directory if it doesn't exist
                                                            import os

                                                            os.makedirs(
                                                                "saved_code",
                                                                exist_ok=True,
                                                            )

                                                            # Save with metadata
                                                            filename = f"saved_code/{save_code_name.strip()}.py"
                                                            metadata = f"""# Generated code saved from Prompt Lab
# Problem: {selected_problem}
# Template: {st.session_state.custom_prompt_name}
# Score: {rr.transform_ran_and_matched_score}/{len(execution_outcomes)}
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
                                                            with open(
                                                                filename, "w"
                                                            ) as f:
                                                                f.write(
                                                                    metadata
                                                                    + code_as_string
                                                                )
                                                            st.success(
                                                                f"✅ Code saved as: {filename}"
                                                            )
                                                        except Exception as e:
                                                            st.error(
                                                                f"❌ Save failed: {e}"
                                                            )
                                                    else:
                                                        st.warning(
                                                            "Please enter a code name"
                                                        )

                                        if exception_message:
                                            st.error(
                                                f"Execution error: {exception_message}"
                                            )
                                    else:
                                        st.warning(
                                            "⚠️ No code block found in LLM response"
                                        )
                                        st.write("**Raw Response Preview:**")
                                        st.text(
                                            full_response[:500] + "..."
                                            if len(full_response) > 500
                                            else full_response
                                        )
                                else:
                                    st.error("❌ No response from LLM")
                            else:
                                st.error("❌ No LLM models configured")

                        except Exception as e:
                            st.error(f"❌ Test failed: {e}")
                            st.code(traceback.format_exc())
                else:
                    st.warning("Please preview the prompt first, then test")

        # Save template
        st.write("### Save Template")
        col_save, col_info = st.columns([1, 2])

        with col_save:
            if st.button(
                "💾 Save Template",
                help="Save the current template to templates/ directory",
            ):
                if st.session_state.custom_prompt_name.strip():
                    try:
                        filename = f"templates/{st.session_state.custom_prompt_name.strip()}.txt"
                        with open(filename, "w") as f:
                            f.write(st.session_state.custom_prompt)
                        st.success(f"✅ Template saved as: {filename}")
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")
                else:
                    st.warning("Please enter a template name")

        with col_info:
            st.info(
                "💡 **Tip**: Saved templates will appear in the LLM Assistant tab for use"
            )

        # Saved Code Management
        st.write("### Saved Generated Code")

        # Check for saved code
        saved_code_files = []
        if Path("saved_code").exists():
            saved_code_files = [f.name for f in Path("saved_code").glob("*.py")]

        if saved_code_files:
            st.write(f"Found {len(saved_code_files)} saved code files:")

            # Create a table of saved code with metadata
            saved_code_data = []
            for filename in saved_code_files:
                try:
                    with open(f"saved_code/{filename}", "r") as f:
                        content = f.read()

                    # Extract metadata from comments
                    lines = content.split("\n")
                    metadata = {}
                    for line in lines[:10]:  # Check first 10 lines for metadata
                        if line.startswith("# Problem:"):
                            metadata["Problem"] = line.replace("# Problem:", "").strip()
                        elif line.startswith("# Template:"):
                            metadata["Template"] = line.replace(
                                "# Template:", ""
                            ).strip()
                        elif line.startswith("# Score:"):
                            metadata["Score"] = line.replace("# Score:", "").strip()
                        elif line.startswith("# Date:"):
                            metadata["Date"] = line.replace("# Date:", "").strip()

                    saved_code_data.append(
                        {
                            "Filename": filename,
                            "Problem": metadata.get("Problem", "Unknown"),
                            "Template": metadata.get("Template", "Unknown"),
                            "Score": metadata.get("Score", "Unknown"),
                            "Date": metadata.get("Date", "Unknown"),
                        }
                    )
                except Exception as e:
                    saved_code_data.append(
                        {
                            "Filename": filename,
                            "Problem": "Error reading",
                            "Template": "Error reading",
                            "Score": "Error reading",
                            "Date": "Error reading",
                        }
                    )

            # Display as dataframe
            df_saved = pd.DataFrame(saved_code_data)
            st.dataframe(df_saved, use_container_width=True)

            # Actions on saved code
            col_load, col_delete, col_view = st.columns([1, 1, 1])

            with col_load:
                selected_to_load = st.selectbox(
                    "Select code to load into editor:",
                    ["Select..."] + saved_code_files,
                    key="load_saved_selector",
                )

                if st.button(
                    "📥 Load to Editor",
                    help="Load selected code into the Code Editor tab",
                ):
                    if selected_to_load != "Select...":
                        try:
                            with open(f"saved_code/{selected_to_load}", "r") as f:
                                saved_code_content = f.read()

                            # Extract just the code part (remove metadata comments)
                            lines = saved_code_content.split("\n")
                            code_start = 0
                            for i, line in enumerate(lines):
                                if line.strip() and not line.strip().startswith("#"):
                                    code_start = i
                                    break

                            clean_code = "\n".join(lines[code_start:])
                            st.session_state.user_code = clean_code
                            st.success(
                                f"✅ Loaded {selected_to_load} into Code Editor!"
                            )
                        except Exception as e:
                            st.error(f"Error loading code: {e}")
                    else:
                        st.warning("Please select a code file")

            with col_view:
                selected_to_view = st.selectbox(
                    "Select code to preview:",
                    ["Select..."] + saved_code_files,
                    key="view_saved_selector",
                )

                if st.button("👁️ Preview Code", help="Preview the selected saved code"):
                    if selected_to_view != "Select...":
                        try:
                            with open(f"saved_code/{selected_to_view}", "r") as f:
                                code_content = f.read()

                            with st.expander(
                                f"Preview: {selected_to_view}", expanded=True
                            ):
                                st.code(code_content, language="python")
                        except Exception as e:
                            st.error(f"Error reading code: {e}")
                    else:
                        st.warning("Please select a code file")

            with col_delete:
                selected_to_delete = st.selectbox(
                    "Select code to delete:",
                    ["Select..."] + saved_code_files,
                    key="delete_saved_selector",
                )

                if st.button(
                    "🗑️ Delete Code", help="Delete the selected saved code file"
                ):
                    if selected_to_delete != "Select...":
                        try:
                            import os

                            os.remove(f"saved_code/{selected_to_delete}")
                            st.success(f"✅ Deleted {selected_to_delete}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting code: {e}")
                    else:
                        st.warning("Please select a code file")

        else:
            st.info(
                "No saved code found. Generate and save some code using the prompt testing above!"
            )

        # Show detailed test results if available
        if st.session_state.get("prompt_test_results"):
            st.write("### Latest Test Results")
            rr, execution_outcomes, exception_message, response, code_as_string = (
                st.session_state.prompt_test_results
            )

            with st.expander("Detailed Test Results", expanded=False):
                # Show extracted code
                st.write("**Extracted Code:**")
                st.code(code_as_string, language="python")

                # Show execution results
                if execution_outcomes:
                    st.write("**Execution Results:**")
                    for i, eo in enumerate(execution_outcomes):
                        st.write(f"Example {i+1}: {'✅' if eo.was_correct else '❌'}")

                # Token usage
                if hasattr(response, "usage") and response.usage:
                    st.write(f"**Token Usage:** {response.usage.total_tokens} tokens")

    with tab5:
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
