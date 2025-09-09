import os
import argparse
import json
import importlib.util
import pdfplumber
import pandas as pd
from typing import Any, Dict
from groq import Groq
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """Extract full text and tables from PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        tables = []
        for page in pdf.pages:
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)  # List of lists of str
        return {'text': text, 'tables': tables}

def get_expected_info(csv_path: str) -> Dict[str, Any]:
    """Get schema and sample from expected CSV."""
    df = pd.read_csv(csv_path)
    return {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample_rows': df.head(3).to_dict('records')
    }

def write_code(target: str, code: str) -> str:
    """Write parser code to file."""
    os.makedirs('custom_parsers', exist_ok=True)
    path = f'custom_parsers/{target}_parser.py'
    with open(path, 'w') as f:
        f.write(code)
    return f"Written to {path}"

def run_test(target: str, pdf_path: str, csv_path: str) -> str:
    """Run internal test: parse PDF and assert equals CSV."""
    parser_path = f'custom_parsers/{target}_parser.py'
    if not os.path.exists(parser_path):
        return "Parser file does not exist."
    spec = importlib.util.spec_from_file_location(f"{target}_parser", parser_path)
    if spec is None:
        return f"Failed to load module spec for {parser_path}."
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        return f"No loader available for {parser_path}."
    try:
        spec.loader.exec_module(module)
    except NameError as e:
        return f"TEST FAILED: Parser module has invalid imports: {str(e)}"
    try:
        parsed_df = module.parse(pdf_path)
        expected_df = pd.read_csv(csv_path)
        if parsed_df.equals(expected_df):
            return "TEST PASSED"
        else:
            return f"TEST FAILED: DataFrames do not match. Parsed shape: {parsed_df.shape}, Expected shape: {expected_df.shape}. Sample diff: Parsed head:\n{parsed_df.head()}\nExpected head:\n{expected_df.head()}"
    except Exception as e:
        return f"TEST FAILED: {str(e)}"

def run_agent(target: str):
    """Main agent loop using Groq tool calling."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    client = Groq(api_key=api_key)
    pdf_path = f"data/{target}/{target}_sample.pdf"
    csv_path = f"data/{target}/{target}_sample.csv"
    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        raise ValueError(f"Sample files not found for {target}. Ensure data/{target}/{target}_sample.pdf and .csv exist.")

    # Tool definitions (JSON schema for Groq)
    tools: list[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_pdf_content",
                "description": "Get raw text and extracted tables from the sample PDF to analyze format.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_expected_info",
                "description": "Get columns, shape, dtypes, and sample rows from the expected CSV.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_parser_code",
                "description": "Write the FULL Python module code (imports + def parse(pdf_path: str) -> pd.DataFrame:). Use pdfplumber for extraction.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Full code string."}},
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_test",
                "description": "Test the current parser against the sample CSV. Returns 'TEST PASSED' or error details.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    ]

    # Tool executors
    def tool_get_pdf_content(tool_call: Any) -> Dict[str, Any]:
        return extract_pdf_content(pdf_path)

    def tool_get_expected_info(tool_call: Any) -> Dict[str, Any]:
        return get_expected_info(csv_path)

    def tool_write_parser_code(tool_call: Any, attempts: int) -> tuple[str, int]:
        attempts += 1
        if attempts > 3:
            return "Max attempts (3) reached. Cannot write more code.", attempts
        try:
            args = json.loads(tool_call.function.arguments)
            code = args.get('code', '')
        except (json.JSONDecodeError, AttributeError) as e:
            return f"Failed to parse tool arguments: {str(e)}", attempts
        write_code(target, code)
        return "Parser code written successfully.", attempts

    def tool_run_test(tool_call: Any) -> str:
        return run_test(target, pdf_path, csv_path)

    tool_map = {
        "get_pdf_content": tool_get_pdf_content,
        "get_expected_info": tool_get_expected_info,
        "write_parser_code": tool_write_parser_code,
        "run_test": tool_run_test
    }

    messages: list[Dict[str, Any]] = [
        {"role": "system", "content": f"""You are a coding agent for writing bank statement PDF parsers.

Target bank: {target}

Goal: Generate {target}_parser.py with:
- Imports: MUST include `import pandas as pd` and `import pdfplumber`
- Function: def parse(pdf_path: str) -> pd.DataFrame
- The DataFrame must match the CSV schema exactly (rows, columns, values, dtypes)

Use tools:
1. Call get_pdf_content to inspect PDF (text/tables for {target}-specific format, e.g., transaction tables).
2. Call get_expected_info for schema (columns like 'Date', 'Description', etc.; match exactly incl. dtypes).
3. Plan: Analyze PDF structure (e.g., extract table from pages, parse dates/debits). Handle {target} quirks (e.g., multi-line descriptions).
4. Generate FULL code string, starting with `import pandas as pd\nimport pdfplumber\n\ndef parse(pdf_path: str) -> pd.DataFrame:\n    ...`. Ensure robust (error handling, all pages processed).
5. Call write_parser_code with the complete code string.
6. Call run_test. If 'TEST PASSED', think "Success!" and FINISH.
7. If test fails, analyze error (e.g., wrong columns, parsing bug), fix code, and rewrite (max 3 attempts). On 3rd fail, FINISH anyway.

Output only tool calls or final thought + 'FINISH'. Keep code clean, typed, documented."""},
        {"role": "user", "content": f"Begin: Write parser for {target} bank statements."}
    ]

    attempts = 0  # Track write attempts
    iteration = 0
    max_iterations = 30  # Safety limit
    print(f"Starting agent for {target}...")

    while iteration < max_iterations:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice="auto",
            temperature=0.1
        )
        message = response.choices[0].message

        # Process tool calls first, if any
        if message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                if func_name in tool_map:
                    try:
                        if not hasattr(tool_call, 'id') or not tool_call.id:
                            print(f"Warning: Tool call for {func_name} missing tool_call_id")
                            messages.append({
                                "role": "tool",
                                "content": f"Error: Tool call for {func_name} missing tool_call_id"
                            })
                            continue
                        if func_name == "write_parser_code":
                            result, attempts = tool_map[func_name](tool_call, attempts)
                        else:
                            result = tool_map[func_name](tool_call)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": json.dumps(result)
                        })
                        print(f"Tool {func_name} executed: {str(result)[:100]}...")
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "unknown",
                            "content": f"Tool error: {str(e)}"
                        })
                        print(f"Tool {func_name} failed: {str(e)}")
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "unknown",
                        "content": f"Unknown tool: {func_name}"
                    })
        else:
            # No tool calls, append assistant message
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            print(f"Agent: {content}")
            if "FINISH" in content.upper():
                final_result = run_test(target, pdf_path, csv_path)
                print(f"Agent complete. Final test: {final_result}")
                if "TEST PASSED" in final_result:
                    print("✅ Parser successful!")
                    return
                else:
                    print("❌ Max attempts reached, but parser written for manual fix.")
                    return

        iteration += 1

    print("Max iterations reached.")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run the bank parser agent.")
    arg_parser.add_argument("--target", required=True, help="Bank name, e.g., icici")
    args = arg_parser.parse_args()
    run_agent(args.target)