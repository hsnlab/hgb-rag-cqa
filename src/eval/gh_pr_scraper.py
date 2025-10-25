from github import Github, RateLimitExceededException, GithubException
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import ast
import hashlib
import pandas as pd

start_datetime = datetime(2025, 8, 5, 9, 30)
TOKEN = "YOUR_TOKEN_HERE"
REPO = "YOUR_REPO_HERE"
PR_NUMS = 3

class FunctionCollector(ast.NodeVisitor):
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.stack = []
        self.functions = {}

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        full_name = '.'.join(self.stack + [node.name])
        start = node.lineno - 1
        end = self._find_end_lineno(node)
        self.functions[full_name] = (start, end)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def _find_end_lineno(self, node):
        max_lineno = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_lineno = max(max_lineno, child.lineno)
        return max_lineno


class GitHubDataCollector:
    def __init__(self, token: str, repo_name: str, after_date: datetime):
        self.token = token
        self.repo_name = repo_name
        self.after_date = after_date

        try:
            self.github_client = Github(self.token)
            self.repo = self.github_client.get_repo(repo_name)
            print(f"Successfully connected to repo: {repo_name}")
        except GithubException as e:
            print(f"[Fatal Error] Could not connect to GitHub or repo {repo_name}: {e}")
            self.repo = None
        except Exception as e:
            print(f"[Fatal Error] Unexpected error during initialization: {e}")
            self.repo = None

    def _get_issue(self, issue_number: int) -> Optional[Any]:
        try:
            return self.repo.get_issue(issue_number)
        except GithubException as e:
            print(f"[Error] Could not fetch issue #{issue_number}: {e}")
            return None

    def _get_pull(self, pr_number: int) -> Optional[Any]:
        try:
            return self.repo.get_pull(pr_number)
        except GithubException as e:
            print(f"[Error] Could not fetch pull request #{pr_number}: {e}")
            return None

    def _hash_function_body(self, lines: List[str]) -> str:
        return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()

    def _extract_function_bodies(self, code: str) -> Dict[str, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"[Warning] Skipping file due to SyntaxError: {e}")
            return {}

        lines = code.splitlines(keepends=True)
        collector = FunctionCollector(lines)
        collector.visit(tree)

        func_bodies = {}
        for name, (start, end) in collector.functions.items():
            # Ensure start and end are within bounds
            start = max(0, start)
            end = min(len(lines), end)
            body_lines = lines[start:end]
            func_bodies[name] = self._hash_function_body(body_lines)

        return func_bodies

    def _get_changed_functions(self, base_code: str, head_code: str) -> List[str]:
        base_funcs = self._extract_function_bodies(base_code)
        head_funcs = self._extract_function_bodies(head_code)

        changed = []
        all_keys = set(base_funcs.keys()) | set(head_funcs.keys())

        for name in all_keys:
            base_hash = base_funcs.get(name)
            head_hash = head_funcs.get(name)
            if base_hash != head_hash:
                changed.append(name)

        return sorted(changed)

    def _get_changed_python_files(self, pr):
        return [
            f for f in pr.get_files()
            if f.filename.endswith(".py") and f.status in {"modified", "renamed"}
        ]

    def _get_file_content(self, repo, path, ref) -> str:
        try:
            content = repo.get_contents(path, ref=ref)
            return content.decoded_content.decode('utf-8')
        except Exception as e:
            print(f"[Warning] Could not get file content for {path}@{ref}: {e}")
            return ""

    def extract_changed_functions_from_pr(self, pr) -> List[str]:

        repo = pr.base.repo
        base_sha = pr.base.sha
        head_sha = pr.head.sha

        changed_funcs = []

        try:
            files = self._get_changed_python_files(pr)
        except Exception as e:
            print(f"[Warning] Could not get files for PR #{pr.number}: {e}")
            return []

        for f in files:
            file_path = f.filename

            base_code = self._get_file_content(repo, file_path, base_sha)
            head_code = self._get_file_content(repo, file_path, head_sha)

            if not base_code or not head_code:
                continue

            try:
                funcs = self._get_changed_functions(base_code, head_code)
                qualified_funcs = [f"{file_path}:{fn}" for fn in funcs]
                changed_funcs.extend(qualified_funcs)
            except Exception as e:
                print(f"[Warning] Error diffing functions in {file_path} for PR #{pr.number}: {e}")

        return sorted(list(set(changed_funcs)))

    def get_linked_prs(self, repo, issue_number):
        issue = self._get_issue(issue_number)
        if not issue:
            return []

        linked_prs = set()

        for event in issue.get_timeline():
            if event.event == "cross-referenced":
                source_issue = event.source.issue
                if source_issue and source_issue.pull_request and source_issue.repository_url == repo.url and source_issue.as_pull_request().state != "closed":
                    linked_prs.add(source_issue.as_pull_request())

        return list(linked_prs)

    def get_linked_issues(self, repo, pr_number):
        issue_for_pr = self._get_issue(pr_number)
        if not issue_for_pr:
            print(f"[Warning] Could not get issue object for PR #{pr_number} to scan timeline.")
            return []

        linked_issues = set()

        try:
            for event in issue_for_pr.get_timeline():
                if event.event == "cross-referenced":
                    source = event.source
                    if source and source.issue and not source.issue.pull_request and source.issue.repository_url == repo.url:
                        linked_issues.add(source.issue)
        except Exception as e:
            print(f"[Warning] Failed to get timeline for PR/issue #{pr_number}: {e}")

        return list(linked_issues)

    def process_pull_request(self, pr_number: int) -> Dict[str, Any]:
        pr = self._get_pull(pr_number)
        if not pr:
            print(f"[Error] Could not fetch PR #{pr_number}. Skipping.")
            return {}

        pr_data = {}

        pr_data['number'] = pr.number
        pr_data['author'] = pr.user.login
        pr_data['created_at'] = pr.created_at
        pr_data['closed_at'] = pr.closed_at
        pr_data['merged_at'] = pr.merged_at
        pr_data['state'] = pr.state
        pr_data['url'] = pr.html_url
        pr_data['title'] = pr.title
        pr_data['body'] = pr.body
        pr_data['pr_problem_statement'] = (pr.title if pr.title else "") + "\n" + (pr.body if pr.body else "")
        pr_data['labels'] = [label.name for label in pr.labels]
        pr_data['additions'] = pr.additions
        pr_data['deletions'] = pr.deletions
        pr_data['changed_files'] = pr.changed_files
        pr_data['comments_count'] = pr.comments
        pr_data['review_comments_count'] = pr.review_comments
        pr_data['commits_count'] = pr.commits

        try:
            comments = pr.get_issue_comments()
            pr_data['comments'] = "\n".join([c.body for c in comments if c.user.type != "Bot"])
        except Exception as e:
            print(f"[Warning] Failed to fetch comments for PR #{pr.number}: {e}")
            pr_data['comments'] = ""

        try:
            pr_data['edit_functions'] = self.extract_changed_functions_from_pr(pr)
        except Exception as e:
            print(f"[Warning] Error extracting changed functions for PR #{pr.number}: {e}")
            pr_data['edit_functions'] = []

        try:
            linked_issues = self.get_linked_issues(self.repo, pr_number)
        except Exception as e:
            print(f"[Warning] Failed to fetch linked issues for PR #{pr_number}: {e}")
            linked_issues = []

        pr_data['linked_issues'] = [issue.html_url for issue in linked_issues]

        issue_problem_statements = []
        issue_comments_list = []

        for issue in linked_issues:
            if issue.user.type == "Bot":
                continue
            try:
                title = issue.title if issue.title else ""
                body = issue.body if issue.body else ""
                issue_problem_statements.append(title + "\n" + body)

                comments = issue.get_comments()
                comments_body = "\n".join([c.body for c in comments if c.user.type != "Bot"])
                if comments_body:
                    issue_comments_list.append(comments_body)

            except Exception as e:
                print(f"[Warning] Error processing linked issue {issue.html_url}: {e}")

        pr_data['issue_problem_statement'] = "\n".join(issue_problem_statements)
        pr_data['issue_comments'] = "\n".join(issue_comments_list)

        return pr_data

    def collect_pull_requests(self, pr_limit: int = 100):
        if not self.repo:
            print("[Fatal Error] Repository object is not initialized. Stopping.")
            return []

        filtered_prs = []

        try:
            print(f"Fetching items from repo since {self.after_date.isoformat()}...")
            all_closed_items = self.repo.get_issues(
                state='closed',
                sort='created',
                direction='desc',
                since=self.after_date
            )

            for item in tqdm(all_closed_items, desc="Filtering and processing PRs..."):
                if len(filtered_prs) >= pr_limit:
                    print(f"\nReached PR limit of {pr_limit}.")
                    break

                if item.created_at.replace(tzinfo=timezone.utc) < self.after_date.replace(tzinfo=timezone.utc):
                    print("\nReached items created before the start date.")
                    break

                if not item.pull_request:
                    continue

                if item.user.type == "Bot" or "bot" in item.user.login.lower():
                    continue

                try:
                    pr_data = self.process_pull_request(item.number)
                    if not pr_data:
                        continue
                    else:
                        filtered_prs.append(pr_data)

                except RateLimitExceededException as e:
                    print(f"[Rate Limit] GitHub API rate limit exceeded: {e}")
                    # save_progress(filtered_issues)
                    print("You may retry after the rate limit resets.")
                    print("Stopping collection due to rate limit.")
                    return filtered_prs
                except GithubException as e:
                    print(f"\n[GitHub Error] PR #{item.number} raised an exception: {e}")
                    continue
                except Exception as e:
                    print(f"\n[Error] Unexpected error with PR #{item.number}: {e}")
                    continue

            print(f"\nSuccessfully collected data for {len(filtered_prs)} pull requests.")
            return filtered_prs

        except RateLimitExceededException as e:
            print(f"[Rate Limit] GitHub API rate limit exceeded: {e}")
            # save_progress(filtered_issues)
            print("You may retry after the rate limit resets.")
            return filtered_prs
        except GithubException as e:
            print(f"[Fatal] GitHub Error. Could not access repo {self.repo_name}: {e}")
            return []
        except Exception as e:
            print(f"[Fatal] Unexpected error: {e}")
            return []


def main():
    print("Starting GitHub PR Scraper...")
    ghdc = GitHubDataCollector(token=TOKEN, repo_name=REPO, after_date=start_datetime)

    collected_prs = ghdc.collect_pull_requests(pr_limit=PR_NUMS)

    if collected_prs:
        df = pd.DataFrame(collected_prs)
        output_filename = "collected_prs.csv"
        df.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
    else:
        print("No data was collected.")


if __name__ == "__main__":
    main()