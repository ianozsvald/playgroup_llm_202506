# playgroup_llm_202506

Ian's playgroup deep dive on LLMs day for 2025-06

**THIS IS A WORK IN PROGRESS, DON'T PICK THIS UP YET**

```
(note Ian first used conda create -n basepython312 python=3.12 to activate a python 3.12 env as 3.9 at least is too old for numpy)
playgroup_llm_202506$ python -m venv ./.venv
playgroup_llm_202506$ . .venv/bin/activate
pip install -r requirements.txt
```
 -am
**NOTE you'll need a `.env` file** which will contain `OPENROUTER_API_KEY=sk-or-v1-...`

# Running the code

## 🆕 Streamlit Web Interface (Recommended)

The easiest way to get started is with the new interactive web interface:

```bash
python run_streamlit.py
```

This will launch a web application that provides:
- 🧩 Visual grid display of ARC puzzles with proper colors
- 💻 Interactive code editor with syntax highlighting
- 🤖 LLM-powered solution generation
- 📊 Real-time execution results and analysis
- 🎯 Side-by-side comparison of expected vs generated outputs

Navigate to `http://localhost:8501` in your browser to access the interface.

## 📟 Command Line Interface (Original)

* `prompt.py` - see how prompts render
* `run_code.py` - execute code on a particular problem
* `method1_text_prompt.py` - run a single prompt many times on 1 problem
* `method2_vllm.py` - try an image representation on 1 problem
* `method3_reflexion.py` - iterate with feedback on 1 problem
* `run_all_problems.py` - run a set of problems once each using a method_

# Tasks

* Run `method0_check_openrouter.py` to check you have a valid `.env` file
* Run `prompt.py`, see how it renders
* Run `run_code.py`, try to run code (see test_run_code.py CODE_3 for a working solution, copy that to a local file)
  * cp CODE_3 into a local file e.g. 'code3_test.py' and use `-c` to pass this in, `/tmp/solution.py` is the default
  * you can try different problems and e.g. broken code
  * ASK yourself - if you have a code block that e.g. does `import scipy` which we don't have, do we report on those kinds of errors? You'll probably want to solve this at some point (maybe not yet)
* Run method1 and try to get to a generalised prompt that can solve multiple problems without change, reliably
  * `python method1_text_prompt.py -t prompt_baseline.txt -p 9565186b -i 3`
    * What does it get wrong? Does it repeat itself?
    * Is it fixating on what it first sees?
  * try with `prompt_baseline_fullclue_956.txt` - but what's the point if we tell it the solution?
  * try with the baseline on 0d3d703e
    * `python method1_text_prompt.py -p 0d3d703e -t prompt_baseline.txt -i 5` how reliable is it?
* Run `run_all_problems.py`, automatically check a set of problems (and try the harder set when you're brave)
* Run `python method2_vllm.py -t prompt_baseline_vllm.txt`, look at the description, this is using a fake image
  * can we make it describe the image well?
  * can we vibe-code an image generator for a specified problem?
  * can we get it to write code?
* Run `python method3_reflexion.py`, does giving feedback help?
