from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from git import Repo
from github import Github
import os
from dotenv import load_dotenv
import shutil
from typing import Annotated, TypedDict

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set GITHUB_TOKEN environment variable")

REPO_NAME = "jaipathak1972/Portfolio-Website"  # Example: your-username/Portfolio-Website

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"  # or use "llama3-70b-8192"
)
class CodeSolution(BaseModel):
    """Schema for code solutions."""
    description: str = Field(description="Description of the solution approach")
    code: str = Field(description="Complete code including imports and docstring")


class GraphState(TypedDict):
    status: Annotated[str, "output"]
    task_content: str
    repo_dir: str
    generation: Annotated[CodeSolution | None, "output"]
    iterations: Annotated[int, "input", "output"]
    test_code: Annotated[str | None, "output"]  
    docs: Annotated[str | None, "output"]
    file_path: Annotated[str | None, "output"]
    anchor: Annotated[str | None, "output"]





import stat

def handle_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repository(github_token: str, repo_name: str) -> tuple[str, str]:
    """Clone the repository and return the local directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.join(script_dir, 'agent-task')

        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, onerror=handle_remove_readonly)  # Windows fix

        g = Github(github_token)
        repo = g.get_repo(repo_name)
        Repo.clone_from(repo.clone_url, repo_dir)
        return repo_dir, ""
    except Exception as e:
        return "", f"Repository setup failed: {str(e)}"

def read_task_file(repo_dir: str) -> tuple[str, str]:
    try:
        task_path = os.path.join(repo_dir, 'tasks', 'task.md')

        # Ask user if they want to add a new task
        add_new = input("📝 Do you want to add a new task to task.md? (yes/no): ").strip().lower()

        if add_new in ["yes", "y"]:
            print("\n📌 Please enter your new task (type END on a new line to finish):")

            # Collect multiline input
            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)

            new_task = "\n".join(lines)
            with open(task_path, "a",encoding= "utf-8") as f:
                f.write(f"\n\n---\n### New Task\n{new_task}\n")

            print("✅ Task added successfully!\n")

        # Now read the full task file content
        with open(task_path, 'r',encoding= "utf-8") as f:
            return f.read(), ""

    except Exception as e:
        return "", f"Failed to read task: {str(e)}"

def generate_solution(state: GraphState):
    if state["status"] == "failed":
        return state

    task_content = state["task_content"]
    iterations = state["iterations"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an AI web developer working with an existing website.

You NEVER generate the full file unless explicitly asked. Instead, return only the **minimal patch** needed for the task.

Always respond in this format:
===DESCRIPTION===
<short summary of the change>
===CODE===
<only the code to insert or replace>
===FILENAME===
<relative path like 'index.html'>
===ANCHOR===
Describe what to search for, like:
- "Replace the <title> tag"
- "Insert this before </body>"
"""),
        ("human", "{task}")
    ])

    try:
        chain = prompt | llm
        print(f"🔄 Generating solution - Attempt #{iterations + 1}")
        raw_output = chain.invoke({"task": task_content}).content

        # Debugging aid (optional):
        # with open("llm_output_debug.txt", "w", encoding="utf-8") as f:
        #     f.write(raw_output)

        # Parse sections
        if "===DESCRIPTION===" not in raw_output or "===CODE===" not in raw_output:
            print("❌ Output format not matched:\n", raw_output)
            return {**state, "status": "failed"}

        description = raw_output.split("===DESCRIPTION===")[1].split("===CODE===")[0].strip()
        code_section = raw_output.split("===CODE===")[1]

        # Handle filename (either ===FILENAME=== or ===FILE===)
        if "===FILENAME===" in code_section:
            code_part, rest = code_section.split("===FILENAME===")
            filename = rest.split("===ANCHOR===")[0].strip()
        elif "===FILE===" in code_section:
            code_part, rest = code_section.split("===FILE===")
            filename = rest.split("===ANCHOR===")[0].strip()
        else:
            print("❌ Missing FILENAME/FILE section.")
            return {**state, "status": "failed"}

        code = code_part.strip()

        # Parse anchor
        if "===ANCHOR===" in raw_output:
            anchor = raw_output.split("===ANCHOR===")[1].strip()
        else:
            anchor = "unknown"

        # 🧼 Sanitize code block
        import re
        code = re.sub(r"^```(?:python|html|css|js)?", "", code)
        code = re.sub(r"```$", "", code).strip()

        return {
            "status": "generated",
            "task_content": task_content,
            "repo_dir": state["repo_dir"],
            "generation": CodeSolution(description=description, code=code),
            "file_path": filename,
            "anchor": anchor,
            "iterations": iterations + 1
        }

    except Exception as e:
        print(f"❌ Generation crashed: {e}")
        return {**state, "status": "failed"}

from bs4 import BeautifulSoup

def patch_file_using_anchor(file_path: str, new_code: str, anchor: str):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    anchor_lower = anchor.lower()
    import re
    # Replace tag content like <h1>, <title>, etc.
    if "replace" in anchor_lower and "<" in anchor_lower and ">" in anchor_lower:
        tag_match = re.search(r"<(\w+)>", anchor_lower)
        if tag_match:
            tag = tag_match.group(1)
            target = soup.find(tag)
            if target:
                # Insert new_code as tag contents (not whole tag)
                new_soup = BeautifulSoup(new_code.strip(), "html.parser")
                target.replace_with(new_soup)
            else:
                raise ValueError(f"⚠️ Tag <{tag}> not found in file")
        else:
            raise ValueError(f"⚠️ Could not extract tag from anchor: {anchor}")

    elif "insert before </body>" in anchor_lower:
        body = soup.find("body")
        if body:
            new_soup = BeautifulSoup(new_code.strip(), "html.parser")
            body.insert_before(new_soup)
        else:
            raise ValueError("⚠️ <body> tag not found for insertion")

    else:
        raise ValueError(f"⚠️ Unknown anchor: {anchor}")

    # Save changes
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(soup.prettify()))
def generate_tests(state: GraphState):
    print("✅ Dummy test logic skipped.")
    return {**state, "status": "tested"}


def test_solution(state: GraphState):
    print("✅ Dummy solution test skipped.")
    return {**state, "status": "tested"}

def create_pr(state: GraphState):
    """Create a pull request with the solution and auto-generated documentation."""
    if state["status"] not in ("tested", "documented") or not state["generation"]:
        return {**state, "status": "failed"}

    try:
        solution = state["generation"]
        repo = Repo(state["repo_dir"])

        # Git branch
        branch_name = f"solution/patch-{state['file_path'].replace('/', '-').replace('.', '-')}"
        current = repo.create_head(branch_name)
        current.checkout()

        # File paths
        solution_path = os.path.join(state["repo_dir"], state["file_path"])
        doc_path = os.path.join(state["repo_dir"], "auto_docs.md")

        # ✅ Use patch logic instead of overwrite
        patch_file_using_anchor(solution_path, solution.code, state["anchor"])

        # Optional: write docs to a markdown file
        docs = state.get("docs", "")
        if docs:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write("# 🧾 Auto-Generated Documentation\n\n" + docs)

        # Git operations
        files_to_commit = [state["file_path"]]
        if docs:
            files_to_commit.append("auto_docs.md")

        repo.index.add(files_to_commit)
        repo.index.commit(f"patch: {solution.description}")
        origin = repo.remote("origin")
        origin.push(branch_name)

        # GitHub PR
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-2:]
        repo_name = '/'.join(repo_name)
        gh_repo = g.get_repo(repo_name)

        pr = gh_repo.create_pull(
            title=f"Auto PR: {solution.description.split('.')[0][:50]}",
            body=f"""### ✨ Auto Patch Summary

**Change**:
{solution.description}

---

### 📘 Documentation:
{docs if docs else "No docs available."}
""",
            base="main",
            head=branch_name
        )

        print(f"✅ Pull Request Created: {pr.html_url}")
        return {**state, "status": "completed", "pr_url": pr.html_url}
    
    except Exception as e:
        print(f"❌ Failed to create PR: {str(e)}")
        return {**state, "status": "failed"}


def should_continue(state: GraphState) -> str:
    print(f"🔁 should_continue → status = {state['status']}, iterations = {state['iterations']}")

    if state["status"] == "failed":
        if state["iterations"] < 3:
            return "generate"
        return "end"

    if state["status"] == "tested":
        return "continue"

    return "end"



def initialize_state(github_token: str, repo_name: str) -> GraphState:
    print("🧪 Initializing state...")
    repo_dir, error = clone_repository(github_token, repo_name)
    if error:
        print(f"❌ Repo clone error: {error}")
        return {
            "status": "failed",
            "task_content": "",
            "repo_dir": "",
            "generation": None,
            "iterations": 0,
            "input_iterations": 0 
        }

    task_content, error = read_task_file(repo_dir)
    if error:
        print(f"❌ Task read error: {error}")
        return {
            "status": "failed",
            "task_content": "",
            "repo_dir": repo_dir,
            "generation": None,
            "iterations": 0
        }

    print("✅ State initialized successfully")
    return {
        "status": "initialized",
        "task_content": task_content,
        "repo_dir": repo_dir,
        "generation": None,
        "iterations": 0
    }
def generate_docs(state: GraphState):
    print("📘 Generating documentation...")

    try:
        solution_code = state.get("generation").code

        if not solution_code:
            print("⚠️ No code found to document.")
            return {**state, "status": "failed"}

        # Prepare prompt
        doc_prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a professional Python documenter. Write a clean and concise docstring or markdown doc explaining the code below."),
            ("human", "{code}")
        ])

        chain = doc_prompt | llm
        response = chain.invoke({"code": solution_code}).content.strip()

        # Save documentation to state
        return {
            **state,
            "docs": response,
            "status": "documented"
        }

    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")
        return {**state, "status": "failed"}


def create_agent(github_token: str, repo_name: str):
    workflow = StateGraph(GraphState)

    # Define all nodes
    workflow.add_node("generate", generate_solution)
    workflow.add_node("generate_tests", generate_tests)
    workflow.add_node("test", test_solution)
    workflow.add_node("generate_docs", generate_docs)
    workflow.add_node("create_pr", create_pr)

    # Define edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "generate_tests")
    workflow.add_edge("generate_tests", "test")

    # Conditional routing after testing
    workflow.add_conditional_edges(
        "test", should_continue,
        {
            "generate": "generate",
            "continue": "generate_docs",
            "end": END
        }
    )

    workflow.add_edge("generate_docs", "create_pr")
    workflow.add_edge("create_pr", END)

    return workflow.compile()




def run_agent(github_token: str, repo_name: str):

    """Run the agent to generate and submit a solution."""
    try:
        agent = create_agent(github_token, repo_name)
        initial_state = initialize_state(github_token, repo_name)

        if initial_state["status"] == "failed":
            print("❌ Failed to initialize agent")
            return {"status": "failed"}

        result = agent.invoke(initial_state)

        if result["status"] == "completed":
            print("🎉 Successfully created PR with solution!")
        else:
            print("❌ Failed to create solution")

        return {
            "status": result["status"],
            "generation": result["generation"].code if result["generation"] else None,
            "pr_url": result.get("pr_url")
        }

    except Exception as e:
        print(f"🔥 Agent execution failed: {e}")
        return {"status": "failed"}


if __name__ == "__main__":
    run_agent(GITHUB_TOKEN, REPO_NAME)
