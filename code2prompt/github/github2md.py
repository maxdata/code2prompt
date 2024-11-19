repo_url = "https://github.com/SciPhi-AI/R2R"

# TODO: index the markdown file to qdrant

import tempfile
import git
# https://gitpython.readthedocs.io/en/stable/quickstart.html

from pathlib import Path
import mimetypes
from typing import Tuple, List
import re
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class GitHubToMarkdown:
    
    def __init__(self, repo_url: str):
        self.repo_url, self.subdir, self.repo_name = self.parse_github_url(repo_url)
    
    def github_to_markdown(self) -> List[str]:
        """
        Clone a GitHub repository and convert its contents to markdown format,
        splitting it into chunks if the token count exceeds the limit.

        Returns:
            List[str]: Paths to the generated markdown files.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository
                try:
                    try:
                        repo = git.Repo.clone_from(self.repo_url, temp_dir, branch='main')
                    except git.exc.GitCommandError:
                        repo = git.Repo.clone_from(self.repo_url, temp_dir, branch='master')
                except Exception as e:
                    print(f"Error cloning repository: {str(e)}")
                    return None
                print(f"Cloned repository to {temp_dir}")
                
                # Process repository
                structure, files = self.process_directory(Path(temp_dir), self.subdir)
                markdown_content = self.generate_content(structure, files, self.repo_name)
                # chunks = self.split_markdown_to_chunks(markdown_content)
                chunks = [markdown_content]

                output_paths = []
                for i, chunk in enumerate(chunks):
                    output_path = f"code/repos/{self.repo_name}_part{i+1}.md"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
                    output_paths.append(output_path)
                
                return output_paths

            except git.exc.GitCommandError as e:
                return f"Error cloning repository: {str(e)}"
            except Exception as e:
                return f"Error processing repository: {str(e)}"

    def convert_notebook_to_python(self, notebook_path: Path) -> str:
        """Convert Jupyter notebook to Python code with markdown cells as comments."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            python_code = []
            cell_counter = 0

            for cell in notebook['cells']:
                cell_counter += 1
                if cell['cell_type'] == 'markdown':
                    markdown_content = ''.join(cell['source'])
                    python_code.extend([
                        f"\n#{'=' * 78}",
                        f"# Markdown Cell {cell_counter}:",
                        f"#{'=' * 78}"
                    ])
                    for line in markdown_content.split('\n'):
                        if line.strip():
                            python_code.append(f"# {line}")
                        else:
                            python_code.append("#")
                elif cell['cell_type'] == 'code':
                    code_content = ''.join(cell['source'])
                    if code_content.strip():
                        python_code.extend([
                            f"\n#{'=' * 78}",
                            f"# Code Cell {cell_counter}:",
                            f"#{'=' * 78}",
                            code_content.rstrip()
                        ])

            return '\n'.join(python_code)

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            return f"# Error converting notebook: {str(e)}"

    def parse_github_url(self, url: str) -> Tuple[str, str, str]:
        """Parse GitHub URL to extract repo URL and subdirectory path."""
        pattern = r'https://github\.com/([^/]+/[^/]+)(?:/(?:tree|blob)/[^/]+)?(/.*)?'
        match = re.match(pattern, url)
        
        if not match:
            raise ValueError("Invalid GitHub URL")
            
        repo_path = match.group(1)
        subdir = (match.group(2) or '').strip('/')
        repo_name = repo_path.split('/')[-1]
        
        repo_url = f'https://github.com/{repo_path}'
        if not repo_url.endswith('.git'):
            repo_url += '.git'
            
        return repo_url, subdir, repo_name

    def count_file_lines(self, file_path: Path) -> int:
        """Count number of lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except (UnicodeDecodeError, OSError):
            return 0

    def is_excluded_file(self, file_path: Path) -> bool:
        """Check if a file should be excluded from content processing."""
        excluded_extensions = {'.html', '.txt'}
        return file_path.suffix in excluded_extensions

    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is a text file based on its mimetype and extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            return file_path.suffix in {'.md', '.txt', '.py', '.js', '.java', '.cpp', 
                                      '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift',
                                      '.kt', '.kts', '.sh', '.bash', '.yaml', '.yml',
                                      '.json', '.xml', '.css', '.scss', '.sql',
                                      '.conf', '.cfg', '.ini', '.toml', '.env',
                                      '.html', '.ipynb'}
        return mime_type.startswith('text/')

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored completely from tree."""
        ignore_patterns = {'.git', '.github', '__pycache__', '.pytest_cache',
                         '.idea', '.vscode', 'node_modules', '.DS_Store'}
        return any(pattern in path.parts for pattern in ignore_patterns)

    def process_directory(self, directory: Path, target_subdir: str = '') -> Tuple[List[str], List[Tuple[str, Path]], set]:
        """Recursively process a directory and return its structure and file paths."""
        structure = []
        files_to_process = []
        included_paths = set()
        base_path = Path(directory)
        
        def _recurse(current_path: Path, level: int = 0):
            indent = "  " * level
            paths = sorted(current_path.iterdir(), 
                         key=lambda x: (not x.is_dir(), x.name.lower()))

            for path in paths:
                if self.should_ignore(path):
                    continue

                rel_path = str(path.relative_to(base_path))
                is_in_target = target_subdir == '' or rel_path.startswith(target_subdir)

                if path.is_dir():
                    line_info = ""
                    prefix = "📂" if is_in_target else "📁"
                    structure.append(f"{indent}- {prefix} **{path.name}/**{line_info}")
                    if is_in_target:
                        included_paths.add(rel_path)
                    _recurse(path, level + 1)
                else:
                    if self.is_text_file(path):
                        line_count = self.count_file_lines(path)
                        line_info = f" `({line_count} lines)`"

                        if is_in_target and not self.is_excluded_file(path):
                            prefix = "📝"
                            included_paths.add(rel_path)
                            files_to_process.append((rel_path, path))
                        else:
                            prefix = "📄"
                            
                        structure.append(f"{indent}- {prefix} {path.name}{line_info}")
                    else:
                        prefix = "📄"
                        structure.append(f"{indent}- {prefix} {path.name}")

        _recurse(base_path)
        return structure, files_to_process

    def generate_content(self, structure: List[str], files: List[Tuple[str, Path]], repo_name: str) -> str:
        """Generate the final markdown content with separate sections."""
        content = [
            f"# {repo_name}\n",
            "## Repository Structure\n",
            "Legend:\n" +
            "- 📂/📝 - Included content in markdown\n" +
            "- 📁/📄 - Excluded or non-text content\n" +
            "```",
            *structure,
            "```\n",
            "## File Contents\n"
        ]

        for rel_path, abs_path in files:
            try:
                if abs_path.suffix == '.ipynb':
                    file_content = self.convert_notebook_to_python(abs_path)
                    content.extend([
                        f"### 📝 `{rel_path}` (converted from notebook)\n",
                        "```python",
                        file_content.rstrip(),
                        "```\n"
                    ])
                else:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                        # Apply the clean-up function
                        cleaned_content = self.clean_file_content(file_content)

                        if cleaned_content.strip():
                            content.extend([
                                f"### 📝 `{rel_path}`\n",
                                f"```{abs_path.suffix[1:] if abs_path.suffix else ''}",
                                cleaned_content.rstrip(),
                                "```\n"
                            ])
            except UnicodeDecodeError:
                content.extend([
                    f"### 📝 `{rel_path}`\n",
                    "*[Binary file or encoding error]*\n"
                ])

        return '\n'.join(content)

    def split_markdown_to_chunks(self, content: str, token_limit: int = 1000000) -> List[str]:
        """
        Split the markdown content into chunks if the token count exceeds the limit, preserving markdown format.

        Args:
            content (str): The markdown content to be split.
            token_limit (int): The maximum number of tokens allowed per chunk.

        Returns:
            List[str]: A list of markdown content chunks.
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        # First split by markdown headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(content)

        # Then further split by character count while respecting header splits
        # chunk_size = token_limit  
        # chunk_overlap = 100  
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # splits = text_splitter.split_documents(md_header_splits)
        
        # Convert back to raw markdown text for each split
        chunks = [document.page_content for document in md_header_splits]

        # Merge chunks if they are smaller than the token limit
        chunks = self.merge_chunks(chunks, token_limit)
        
        return chunks

    def merge_chunks(self, chunks: List[str], token_limit: int) -> List[str]:
        """Merge chunks if their combined token count is below the limit."""
        merged_chunks = []
        current_chunk = []

        for chunk in chunks:
            # Calculate the current token size
            current_size = len(' '.join(current_chunk).split())
            chunk_size = len(chunk.split())

            if current_size + chunk_size <= token_limit:
                current_chunk.append(chunk)
            else:
                merged_chunks.append(' '.join(current_chunk))
                current_chunk = [chunk]

        # Add the last chunk if it exists
        if current_chunk:
            merged_chunks.append(' '.join(current_chunk))
        
        return merged_chunks

    def clean_file_content(self, content: str) -> str:
        """
        Clean up file content by removing unnecessary whitespace and comments
        to reduce the number of tokens in the markdown.

        Args:
            content (str): The original content of the file.

        Returns:
            str: The cleaned content.
        """
        # Remove unnecessary whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove comments (for some common languages, e.g., Python, JavaScript, etc.)
        # This will need to be adapted for different comment styles if necessary
        cleaned_lines = []
        for line in content.split('\n'):
            if not line.strip().startswith('#') and not line.strip().startswith('//'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

# Example usage
if __name__ == "__main__":    
    converter = GitHubToMarkdown(repo_url)
    output_files = converter.github_to_markdown()
    for file in output_files:
        print(f"Markdown file generated: {file}")
