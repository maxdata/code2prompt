# Table of Contents
- code2prompt/config.py
- code2prompt/__init__.py
- code2prompt/main.py
- code2prompt/core/template_processor.py
- code2prompt/core/generate_content.py
- code2prompt/core/process_files.py
- code2prompt/core/__init__.py
- code2prompt/core/file_path_retriever.py
- code2prompt/core/process_file.py
- code2prompt/core/write_output.py
- code2prompt/utils/is_ignored.py
- code2prompt/utils/count_tokens.py
- code2prompt/utils/config.py
- code2prompt/utils/is_binary.py
- code2prompt/utils/output_utils.py
- code2prompt/utils/analyzer.py
- code2prompt/utils/create_template_directory.py
- code2prompt/utils/generate_markdown_content.py
- code2prompt/utils/include_loader.py
- code2prompt/utils/is_filtered.py
- code2prompt/utils/price_calculator.py
- code2prompt/utils/file_utils.py
- code2prompt/utils/get_gitignore_patterns.py
- code2prompt/utils/add_line_numbers.py
- code2prompt/utils/should_process_file.py
- code2prompt/utils/display_price_table.py
- code2prompt/utils/language_inference.py
- code2prompt/utils/logging_utils.py
- code2prompt/utils/parse_gitignore.py
- code2prompt/comment_stripper/strip_comments.py
- code2prompt/comment_stripper/sql_style.py
- code2prompt/comment_stripper/c_style.py
- code2prompt/comment_stripper/python_style.py
- code2prompt/comment_stripper/__init__.py
- code2prompt/comment_stripper/shell_style.py
- code2prompt/comment_stripper/r_style.py
- code2prompt/comment_stripper/matlab_style.py
- code2prompt/comment_stripper/html_style.py
- code2prompt/templates/create-readme.j2
- code2prompt/templates/default.j2
- code2prompt/templates/improve-this-prompt.j2
- code2prompt/templates/create-function.j2
- code2prompt/templates/analyze-code.j2
- code2prompt/templates/code-review.j2
- code2prompt/commands/interactive_selector.py
- code2prompt/commands/generate.py
- code2prompt/commands/__init__.py
- code2prompt/commands/base_command.py
- code2prompt/commands/analyze.py
- code2prompt/data/token_price.json

## File: code2prompt/config.py

- Extension: .py
- Language: python
- Size: 5211 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/config.py

from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, ValidationError

class Configuration(BaseModel):
    """
    Configuration class for code2prompt tool.
    
    This class uses Pydantic for data validation and settings management.
    It defines all the configuration options available for the code2prompt tool.
    """

    path: List[Path] = Field(default_factory=list, description="Path(s) to the directory or file to process.")
    output: Optional[Path] = Field(None, description="Name of the output Markdown file.")
    gitignore: Optional[Path] = Field(None, description="Path to the .gitignore file.")
    filter: Optional[str] = Field(None, description="Comma-separated filter patterns to include files.")
    exclude: Optional[str] = Field(None, description="Comma-separated patterns to exclude files.")
    case_sensitive: bool = Field(False, description="Perform case-sensitive pattern matching.")
    suppress_comments: bool = Field(False, description="Strip comments from the code files.")
    line_number: bool = Field(False, description="Add line numbers to source code blocks.")
    no_codeblock: bool = Field(False, description="Disable wrapping code inside markdown code blocks.")
    template: Optional[Path] = Field(None, description="Path to a Jinja2 template file for custom prompt generation.")
    tokens: bool = Field(False, description="Display the token count of the generated prompt.")
    encoding: str = Field("cl100k_base", description="Specify the tokenizer encoding to use.")
    create_templates: bool = Field(False, description="Create a templates directory with example templates.")
    log_level: str = Field("INFO", description="Set the logging level.")
    price: bool = Field(False, description="Display the estimated price of tokens based on provider and model.")
    provider: Optional[str] = Field(None, description="Specify the provider for price calculation.")
    model: Optional[str] = Field(None, description="Specify the model for price calculation.")
    output_tokens: int = Field(1000, description="Specify the number of output tokens for price calculation.")
    analyze: bool = Field(False, description="Analyze the codebase and provide a summary of file extensions.")
    format: str = Field("flat", description="Format of the analysis output (flat or tree-like).")
    interactive: bool = Field(False, description="Interactive mode to select files.")
    
    # Add the syntax_map attribute
    syntax_map: Dict[str, str] = Field(default_factory=dict, description="Custom syntax mappings for language inference.")

    @field_validator('encoding')
    @classmethod
    def validate_encoding(cls, v: str) -> str:
        valid_encodings = ["cl100k_base", "p50k_base", "p50k_edit", "r50k_base"]
        if v not in valid_encodings:
            raise ValueError(f"Invalid encoding. Must be one of: {', '.join(valid_encodings)}")
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator('format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["flat", "tree"]
        if v not in valid_formats:
            raise ValueError(f"Invalid format. Must be one of: {', '.join(valid_formats)}")
        return v

    @classmethod
    def load_from_file(cls, file_path: Path) -> "Configuration":
        """
        Load configuration from a file.

        Args:
            file_path (Path): Path to the configuration file.

        Returns:
            Configuration: Loaded configuration object.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValidationError: If the configuration file is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with file_path.open() as f:
                config_data = f.read()
            return cls.model_validate_json(config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration file: {e}")

    def merge(self, cli_options: dict) -> "Configuration":
        """
        Merge CLI options with the current configuration.

        Args:
            cli_options (dict): Dictionary of CLI options.

        Returns:
            Configuration: New configuration object with merged options.
        """
        # Create a new dict with all current config values
        merged_dict = self.model_dump()

        # Update with CLI options, but only if they're different from the default
        for key, value in cli_options.items():
            if value is not None and value != self.model_fields[key].default:
                merged_dict[key] = value

        # Create a new Configuration object with the merged options
        return Configuration.model_validate(merged_dict)
```

## File: code2prompt/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python

```

## File: code2prompt/main.py

- Extension: .py
- Language: python
- Size: 2729 bytes
- Created: 2024-11-18 22:28:28
- Modified: 2024-11-18 22:28:28

### Code

```python
"""Main module for the code2prompt CLI tool."""

import logging
from pathlib import Path
from code2prompt.commands.analyze import AnalyzeCommand
from code2prompt.commands.generate import GenerateCommand
from code2prompt.utils.logging_utils import setup_logger
from code2prompt.commands.interactive_selector import InteractiveFileSelector
from code2prompt.core.file_path_retriever import retrieve_file_paths


def generate(ctx, path=(), syntax_map=None, output=None, **options):
    """Generate markdown from code files"""

    # Parse the syntax_map option into a dictionary
    if syntax_map:
        syntax_map_dict = {'.' + ext.strip(): syntax.strip() for ext, syntax in 
                           (mapping.split(':') for mapping in syntax_map.split(','))}
        options['syntax_map'] = syntax_map_dict  # Replace the string with the dictionary

    config = ctx['config'].merge(options)
    logger = setup_logger(level=config.log_level)

    selected_paths = [Path(p) for p in config.path]

    # Check if selected_paths is empty before proceeding
    if not selected_paths:
        logging.error("No file paths provided. Please specify valid paths.")
        return  # Exit the function if no paths are provided

    filter_patterns = config.filter.split(",") if config.filter else []
    exclude_patterns = config.exclude.split(",") if config.exclude else []
    case_sensitive = config.case_sensitive
    gitignore = config.gitignore

    # Handle both directory and file inputs
    filtered_paths = []
    for path in selected_paths:
        if path.is_dir():
            filtered_paths.extend(retrieve_file_paths(
                file_paths=[path],
                gitignore=gitignore,
                filter_patterns=filter_patterns,
                exclude_patterns=exclude_patterns,
                case_sensitive=case_sensitive,
            ))
        elif path.is_file():
            filtered_paths.append(path)

    if filtered_paths and config.interactive:
        file_selector = InteractiveFileSelector(filtered_paths, filtered_paths)
        filtered_selected_path = file_selector.run()
        config.path = filtered_selected_path
    else:
        config.path = filtered_paths

    command = GenerateCommand(config, logger)
    command.execute()

    logger.info("Markdown generation completed.")
    

def analyze(ctx, path=(), format='flat'):
    """Analyze codebase structure"""
    config = ctx['config'].merge({'path': path, 'format': format})
    logger = setup_logger(level=config.log_level)
    logger.info("Analyzing codebase with options: %s", {'path': path, 'format': format})

    command = AnalyzeCommand(config, logger)
    command.execute()

    logger.info("Codebase analysis completed.")

```

## File: code2prompt/core/template_processor.py

- Extension: .py
- Language: python
- Size: 3842 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import os
from jinja2 import Environment
from jinja2 import TemplateNotFound
from code2prompt.utils.include_loader import CircularIncludeError, IncludeLoader
from code2prompt.utils.logging_utils import log_error
from prompt_toolkit import prompt
import re


def load_template(template_path):
    """
    Load a template file from the given path.

    Args:
        template_path (str): The path to the template file.

    Returns:
        str: The contents of the template file.

    Raises:
        IOError: If there is an error loading the template file.
    """
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error loading template file: {e}") from e


def get_user_inputs(template_content):
    """
    Extracts user inputs from a template content.

    Args:
        template_content (str): The content of the template.

    Returns:
        dict: A dictionary containing the user inputs, where the keys are the variable names and the values are the user-entered values.
    """
    pattern = r"{{\s*input:([^{}]+?)\s*}}"
    matches = re.finditer(pattern, template_content)

    user_inputs = {}
    for match in matches:
        var_name = match.group(1).strip()
        if var_name and var_name not in user_inputs:
            user_inputs[var_name] = prompt(f"Enter value for {var_name}: ")

    return user_inputs


def replace_input_placeholders(template_content, user_inputs):
    """
    Replaces input placeholders in the template content with user inputs.

    Args:
        template_content (str): The content of the template.
        user_inputs (dict): A dictionary containing user inputs.

    Returns:
        str: The template content with input placeholders replaced by user inputs.
    """
    pattern = r"{{\s*input:([^{}]+?)\s*}}"

    def replace_func(match):
        var_name = match.group(1).strip()
        return user_inputs.get(var_name, "")

    return re.sub(pattern, replace_func, template_content)


def process_template(template_content, files_data, user_inputs, template_path):
    """
    Process a template by replacing input placeholders with user-provided values and rendering the template.

    Args:
        template_content (str): The content of the template to be processed.
        files_data (dict): A dictionary containing data for files that may be referenced in the template.
        user_inputs (dict): A dictionary containing user-provided values for input placeholders in the template.
        template_path (str): The path to the template file.

    Returns:
        str: The processed template content with input placeholders replaced and rendered.

    Raises:
        TemplateNotFound: If the template file is not found at the specified path.
        CircularIncludeError: If a circular include is detected in the template.
        Exception: If there is an error processing the template.

    """
    try:
        template_dir = os.path.dirname(template_path)
        env = Environment(
            loader=IncludeLoader(template_dir),
            autoescape=True,
            keep_trailing_newline=True,
        )
        # Replace input placeholders with user-provided values
        processed_content = replace_input_placeholders(template_content, user_inputs)
        template = env.from_string(processed_content)
        return template.render(files=files_data, **user_inputs)
    except TemplateNotFound as e:
        log_error(
            f"Template file not found: {e.name}. Please check the path and ensure the file exists."
        )
        return None
    except CircularIncludeError as e:
        log_error(f"Circular include detected: {str(e)}")
        return None
    except IOError as e:
        log_error(f"Error processing template: {e}")
        return None

```

## File: code2prompt/core/generate_content.py

- Extension: .py
- Language: python
- Size: 1237 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from code2prompt.core.template_processor import get_user_inputs, load_template, process_template
from code2prompt.utils.generate_markdown_content import generate_markdown_content


def generate_content(files_data, options):
    """
    Generate content based on the provided files data and options.

    This function either processes a Jinja2 template with the given files data and user inputs
    or generates markdown content directly from the files data, depending on whether a
    template option is provided.

    Args:
        files_data (list): A list of dictionaries containing processed file data.
        options (dict): A dictionary containing options such as template path and whether
                        to wrap code inside markdown code blocks.

    Returns:
        str: The generated content as a string, either from processing a template or
             directly generating markdown content.
    """
    if options['template']:
        template_content = load_template(options['template'])
        user_inputs = get_user_inputs(template_content)
        return process_template(template_content, files_data, user_inputs, options['template'])
    return generate_markdown_content(files_data, options['no_codeblock'])
```

## File: code2prompt/core/process_files.py

- Extension: .py
- Language: python
- Size: 1363 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains functions for processing files and directories.
"""

from pathlib import Path
from typing import List, Dict, Any
from code2prompt.core.process_file import process_file


def process_files(
    file_paths: List[Path],
    line_number: bool,
    no_codeblock: bool,
    suppress_comments: bool,
    syntax_map: dict  # Add this parameter
) -> List[Dict[str, Any]]:
    """
    Processes files or directories based on the provided paths.

    Args:
    options (dict): A dictionary containing options such as paths, gitignore patterns,
                    and flags for processing files.

    Returns:
    list: A list of dictionaries containing processed file data.
    """
    files_data = []
    
    # Test file paths if List[Path] type
    if not (isinstance(file_paths, list) and all(isinstance(path, Path) for path in file_paths)): 
        raise ValueError("file_paths must be a list of Path objects")

    # Use get_file_paths to retrieve all file paths to process
    for path in file_paths:
        result = process_file(
            file_path=path,
            suppress_comments=suppress_comments,
            line_number=line_number,
            no_codeblock=no_codeblock,
            syntax_map=syntax_map  # Ensure this is being passed
        )
        if result:
            files_data.append(result)

    return files_data

```

## File: code2prompt/core/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python

```

## File: code2prompt/core/file_path_retriever.py

- Extension: .py
- Language: python
- Size: 2254 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains the function to get file paths based on the provided options.
"""

from pathlib import Path
from code2prompt.utils.get_gitignore_patterns import get_gitignore_patterns
from code2prompt.utils.should_process_file import should_process_file


def retrieve_file_paths(
    file_paths: list[Path],
    filter_patterns: list[str],
    exclude_patterns: list[str],
    case_sensitive: bool,
    gitignore: list[str],
) -> list[Path]:
    """
    Retrieves file paths based on the provided options.

    Args:
    file_paths (list[Path]): A list of paths to retrieve.
    filter_patterns (list[str]): Patterns to include.
    exclude_patterns (list[str]): Patterns to exclude.
    case_sensitive (bool): Whether the filtering should be case sensitive.
    gitignore (list[str]): Gitignore patterns to consider.

    Returns:
    list[Path]: A list of file paths that should be processed.
    """
    if not file_paths:
        raise ValueError("file_paths list cannot be empty.")

    retrieved_paths: list[Path] = []

    for path in file_paths:
        try:
            path = Path(path)

            # Get gitignore patterns for the current path
            gitignore_patterns = get_gitignore_patterns(
                path.parent if path.is_file() else path, gitignore
            )

            # Add the top-level directory if it should be processed
            if path.is_dir() and should_process_file(
                path,
                gitignore_patterns,
                path.parent,
                filter_patterns,
                exclude_patterns,
                case_sensitive,
            ):
                retrieved_paths.append(path)

            # Add files and directories within the top-level directory
            for file_path in path.rglob("*"):
                if should_process_file(
                    file_path,
                    gitignore_patterns,
                    path,
                    filter_patterns,
                    exclude_patterns,
                    case_sensitive,
                ):
                    retrieved_paths.append(file_path)

        except (FileNotFoundError, PermissionError) as e:
            print(f"Error processing path {path}: {e}")

    return retrieved_paths
```

## File: code2prompt/core/process_file.py

- Extension: .py
- Language: python
- Size: 2130 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains the function to process a file and extract its metadata and content.
"""

from pathlib import Path
from datetime import datetime

from code2prompt.utils.add_line_numbers import add_line_numbers
from code2prompt.utils.language_inference import infer_language
from code2prompt.comment_stripper.strip_comments import strip_comments


def process_file(
    file_path: Path, suppress_comments: bool, line_number: bool, no_codeblock: bool, syntax_map: dict
):
    """
    Processes a given file to extract its metadata and content.

    Parameters:
    - file_path (Path): The path to the file to be processed.
    - suppress_comments (bool): Flag indicating whether to remove comments from the file content.
    - line_number (bool): Flag indicating whether to add line numbers to the file content.
    - no_codeblock (bool): Flag indicating whether to disable wrapping code inside markdown code blocks.
    - syntax_map (dict): Custom syntax mappings for language inference.

    Returns:
    dict: A dictionary containing the file information and content.
    """
    file_extension = file_path.suffix
    file_size = file_path.stat().st_size
    file_creation_time = datetime.fromtimestamp(file_path.stat().st_ctime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    file_modification_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:
        with file_path.open("r", encoding="utf-8") as f:
            file_content = f.read()

        language = infer_language(file_path.name, syntax_map)

        if suppress_comments and language != "unknown":
            file_content = strip_comments(file_content, language)

        if line_number:
            file_content = add_line_numbers(file_content)
    except UnicodeDecodeError:
        return None

    return {
        "path": str(file_path),
        "extension": file_extension,
        "language": language,
        "size": file_size,
        "created": file_creation_time,
        "modified": file_modification_time,
        "content": file_content,
        "no_codeblock": no_codeblock,
    }

```

## File: code2prompt/core/write_output.py

- Extension: .py
- Language: python
- Size: 1451 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from pathlib import Path
import click
import pyperclip
from code2prompt.utils.logging_utils import (
    log_output_created,
    log_error,
    log_clipboard_copy,
    log_token_count,
)

def write_output(content, output_path, copy_to_clipboard=True, token_count=None):
    """
    Writes the generated content to a file or prints it to the console,
    logs the token count if provided, and copies the content to the clipboard.

    Parameters:
    - content (str): The content to be written, printed, and copied.
    - output_path (str): The path to the file where the content should be written.
                         If None, the content is printed to the console.
    - copy_to_clipboard (bool): Whether to copy the content to the clipboard.
    - token_count (int, optional): The number of tokens in the content.

    Returns: None
    """
    if output_path:
        try:
            with Path(output_path).open("w", encoding="utf-8") as output_file:
                output_file.write(content)
            log_output_created(output_path)
        except IOError as e:
            log_error(f"Error writing to output file: {e}")
    else:
        click.echo(content)

    if token_count is not None:
        log_token_count(token_count)

    if copy_to_clipboard:
        try:
            pyperclip.copy(content)
            log_clipboard_copy(success=True)
        except Exception as _e:
            log_clipboard_copy(success=False)
        
```

## File: code2prompt/utils/is_ignored.py

- Extension: .py
- Language: python
- Size: 1170 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from fnmatch import fnmatch
from pathlib import Path



def is_ignored(file_path: Path, gitignore_patterns: list, base_path: Path) -> bool:
    """
    Check if a file is ignored based on gitignore patterns.

    Args:
        file_path (Path): The path of the file to check.
        gitignore_patterns (list): List of gitignore patterns.
        base_path (Path): The base path to resolve relative paths.

    Returns:
        bool: True if the file is ignored, False otherwise.
    """
    relative_path = file_path.relative_to(base_path)
    for pattern in gitignore_patterns:
        pattern = pattern.rstrip("/")
        if pattern.startswith("/"):
            if fnmatch(str(relative_path), pattern[1:]):
                return True
            if fnmatch(str(relative_path.parent), pattern[1:]):
                return True
        else:
            for path in relative_path.parents:
                if fnmatch(str(path / relative_path.name), pattern):
                    return True
                if fnmatch(str(path), pattern):
                    return True
            if fnmatch(str(relative_path), pattern):
                return True
    return False
```

## File: code2prompt/utils/count_tokens.py

- Extension: .py
- Language: python
- Size: 571 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import click
import tiktoken


def count_tokens(text: str, encoding: str) -> int:
    """
    Count the number of tokens in the given text using the specified encoding.

    Args:
        text (str): The text to tokenize and count.
        encoding (str): The encoding to use for tokenization.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoder = tiktoken.get_encoding(encoding)
        return len(encoder.encode(text))
    except Exception as e:
        click.echo(f"Error counting tokens: {str(e)}", err=True)
        return 0
```

## File: code2prompt/utils/config.py

- Extension: .py
- Language: python
- Size: 1982 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/config.py
import json
from pathlib import Path

def load_config(current_dir):
    """
    Load configuration from .code2promptrc files.
    Searches in the current directory and all parent directories up to the home directory.
    """
    config = {}
    current_path = Path(current_dir).resolve()
    home_path = Path.home()
    while current_path >= home_path:
        rc_file = current_path / '.code2promptrc'
        if rc_file.is_file():
            with open(rc_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                if 'path' in file_config and isinstance(file_config['path'], str):
                    file_config['path'] = file_config['path'].split(',')
                config.update(file_config)
        if current_path == home_path:
            break
        current_path = current_path.parent
    return config

def merge_options(cli_options: dict, config_options: dict, default_options: dict) -> dict:
    """
    Merge CLI options, config options, and default options.
    CLI options take precedence over config options, which take precedence over default options.
    """
    merged = default_options.copy()
    
    # Update with config options
    for key, value in config_options.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_options({}, value, merged[key])
        else:
            merged[key] = value
    
    # Update with CLI options, but only if they're different from the default
    for key, value in cli_options.items():
        if value != default_options.get(key):
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_options(value, {}, merged[key])
            else:
                merged[key] = value
    
    # Special handling for 'path'
    if not merged['path'] and 'path' in config_options:
        merged['path'] = config_options['path']
    
    return merged
```

## File: code2prompt/utils/is_binary.py

- Extension: .py
- Language: python
- Size: 261 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
def is_binary(file_path):
    try:
        with open(file_path, "rb") as file:
            chunk = file.read(1024)
            return b"\x00" in chunk
    except IOError:
        print(f"Error: The file at {file_path} could not be opened.")
        return False
```

## File: code2prompt/utils/output_utils.py

- Extension: .py
- Language: python
- Size: 4309 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/utils/output_utils.py

from pathlib import Path
import logging
from typing import Dict, List, Optional

from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax

from code2prompt.config import Configuration


def generate_content(files_data: List[Dict], config: Configuration) -> str:
    """
    Generate content based on the provided files data and configuration.

    Args:
        files_data (List[Dict]): A list of dictionaries containing processed file data.
        config (Configuration): Configuration object containing options.

    Returns:
        str: The generated content as a string.
    """
    if config.template:
        return _process_template(files_data, config)
    return _generate_markdown_content(files_data, config.no_codeblock)


def _process_template(files_data: List[Dict], config: Configuration) -> str:
    """
    Process a Jinja2 template with the given files data and user inputs.

    Args:
        files_data (List[Dict]): A list of dictionaries containing processed file data.
        config (Configuration): Configuration object containing options.

    Returns:
        str: The processed template content.
    """
    from code2prompt.core.template_processor import (
        get_user_inputs,
        load_template,
        process_template,
    )

    template_content = load_template(config.template)
    user_inputs = get_user_inputs(template_content)
    return process_template(template_content, files_data, user_inputs, config.template)


def _generate_markdown_content(files_data: List[Dict], no_codeblock: bool) -> str:
    """
    Generate markdown content from the provided files data.

    Args:
        files_data (List[Dict]): A list of dictionaries containing file information and content.
        no_codeblock (bool): Flag indicating whether to disable wrapping code inside markdown code blocks.

    Returns:
        str: A Markdown-formatted string containing the table of contents and the file contents.
    """
    table_of_contents = [f"- {file['path']}\n" for file in files_data]
    content = []

    for file in files_data:
        file_info = (
            f"## File: {file['path']}\n\n"
            f"- Extension: {file['extension']}\n"
            f"- Language: {file['language']}\n"
            f"- Size: {file['size']} bytes\n"
            f"- Created: {file['created']}\n"
            f"- Modified: {file['modified']}\n\n"
        )

        if no_codeblock:
            file_code = f"### Code\n\n{file['content']}\n\n"
        else:
            file_code = f"### Code\n\n```{file['language']}\n{file['content']}\n```\n\n"

        content.append(file_info + file_code)

    return "# Table of Contents\n" + "".join(table_of_contents) + "\n" + "".join(content)


def write_output(content: str, output_path: Optional[Path], logger: logging.Logger):
    """
    Write the generated content to a file or print it to the console.

    Args:
        content (str): The content to be written or printed.
        output_path (Optional[Path]): The path to the file where the content should be written.
                                      If None, the content is printed to the console.
        logger (logging.Logger): Logger instance for logging messages.
    """
    if output_path:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write(content)
            logger.info(f"Output file created: {output_path}")
        except IOError as e:
            logger.error(f"Error writing to output file: {e}")
    else:
        rprint(Panel(Syntax(content, "markdown", theme="monokai", line_numbers=True)))


def log_token_count(count: int):
    """
    Log the total number of tokens processed.

    Args:
        count (int): The total number of tokens processed.
    """
    rprint(f"[cyan]🔢 Token count: {count}[/cyan]")


def log_clipboard_copy(success: bool = True):
    """
    Log whether the content was successfully copied to the clipboard.

    Args:
        success (bool): Indicates whether the content was successfully copied to the clipboard.
    """
    if success:
        rprint("[green]📋 Content copied to clipboard[/green]")
    else:
        rprint("[yellow]📋 Failed to copy content to clipboard[/yellow]")
```

## File: code2prompt/utils/analyzer.py

- Extension: .py
- Language: python
- Size: 2988 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_codebase(path: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Analyze the codebase and return file extension information.
    
    Args:
        path (str): The path to the codebase directory.
    
    Returns:
        Tuple[Dict[str, int], Dict[str, List[str]]]: A tuple containing:
            - A dictionary of file extensions and their counts.
            - A dictionary of file extensions and the directories containing them.
    """
    extension_counts = defaultdict(int)
    extension_dirs = defaultdict(set)
    
    file_count = 0
    for file_path in Path(path).rglob('*'):
        if file_path.is_file():
            file_count += 1
            ext = file_path.suffix.lower()
            if ext:
                extension_counts[ext] += 1
                extension_dirs[ext].add(str(file_path.parent))
    
    if file_count == 0:
        return {"No files found": 0}, {}
    
    return dict(extension_counts), {k: list(v) for k, v in extension_dirs.items()}
    

def format_flat_output(extension_counts: Dict[str, int]) -> str:
    """
    Format the analysis results in a flat structure.
    
    Args:
        extension_counts (Dict[str, int]): A dictionary of file extensions and their counts.
    
    Returns:
        str: Formatted output string.
    """
    output = []
    for ext, count in sorted(extension_counts.items()):
        output.append(f"{ext}: {count} file{'s' if count > 1 else ''}")
    return "\n".join(output)

def format_tree_output(extension_dirs: Dict[str, List[str]]) -> str:
    """
    Format the analysis results in a tree-like structure.
    
    Args:
        extension_dirs (Dict[str, List[str]]): A dictionary of file extensions and their directories.
    
    Returns:
        str: Formatted output string.
    """
    def format_tree(node, prefix=""):
        output = []
        for i, (key, value) in enumerate(node.items()):
            is_last = i == len(node) - 1
            output.append(f"{prefix}{'└── ' if is_last else '├── '}{key}")
            if isinstance(value, dict):
                extension = "    " if is_last else "│   "
                output.extend(format_tree(value, prefix + extension))
        return output

    tree = {}
    for ext, dirs in extension_dirs.items():
        for dir_path in dirs:
            current = tree
            for part in Path(dir_path).parts:
                current = current.setdefault(part, {})
            current[ext] = {}

    return "\n".join(format_tree(tree))

def get_extension_list(extension_counts: Dict[str, int]) -> str:
    """
    Generate a comma-separated list of file extensions.
    
    Args:
        extension_counts (Dict[str, int]): A dictionary of file extensions and their counts.
    
    Returns:
        str: Comma-separated list of file extensions.
    """
    return ",".join(sorted(extension_counts.keys()))
```

## File: code2prompt/utils/create_template_directory.py

- Extension: .py
- Language: python
- Size: 4239 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import os
import shutil
from pathlib import Path
import logging
import tempfile

logger = logging.getLogger(__name__)

def create_templates_directory(package_templates_dir: Path, templates_dir: Path, dry_run=False, force=False, skip_existing=False):
    """
    Create a 'templates' directory in the current working directory and populate it with template files
    from the package's templates directory.

    Args:
        package_templates_dir (Path): Path to the package's templates directory.
        templates_dir (Path): Path to the directory where templates will be copied.
        dry_run (bool): If True, show what changes would be made without making them.
        force (bool): If True, overwrite existing files without prompting.
        skip_existing (bool): If True, skip existing files without prompting or overwriting.

    Raises:
        FileNotFoundError: If the package templates directory is not found.
        PermissionError: If there's a permission issue creating directories or copying files.
        IOError: If there's an IO error during the copy process.
    """
    if not package_templates_dir.exists():
        logger.error(f"Package templates directory not found: {package_templates_dir}")
        raise FileNotFoundError(f"Package templates directory not found: {package_templates_dir}")

    if dry_run:
        logger.info("Dry run mode: No changes will be made.")

    try:
        if not dry_run:
            templates_dir.mkdir(exist_ok=True, parents=True)
            if not os.access(templates_dir, os.W_OK):
                raise PermissionError(f"No write permission for directory: {templates_dir}")
        logger.info(f"Templates directory {'would be' if dry_run else 'was'} created at: {templates_dir}")
    except PermissionError as e:
        logger.error(f"Permission error: {str(e)}")
        raise

    # Check available disk space only if not in dry run mode
    if not dry_run:
        try:
            _, _, free = shutil.disk_usage(templates_dir)
            required_space = sum(f.stat().st_size for f in package_templates_dir.glob('**/*') if f.is_file())
            if free < required_space:
                raise IOError(f"Insufficient disk space. Required: {required_space}, Available: {free}")
        except OSError as e:
            logger.error(f"Error checking disk space: {str(e)}")
            raise

    copied_files = []
    try:
        for template_file in package_templates_dir.iterdir():
            if template_file.is_file():
                dest_file = templates_dir / template_file.name
                if dest_file.exists():
                    if skip_existing:
                        logger.info(f"Skipping existing file: {dest_file}")
                        continue
                    if not force:
                        if dry_run:
                            logger.info(f"Would prompt to overwrite: {dest_file}")
                            continue
                        overwrite = input(f"{dest_file} already exists. Overwrite? (y/n): ").lower() == 'y'
                        if not overwrite:
                            logger.info(f"Skipping: {template_file.name}")
                            continue

                try:
                    if not dry_run:
                        # Use a temporary file to ensure atomic write
                        with tempfile.NamedTemporaryFile(dir=templates_dir, delete=False) as tmp_file:
                            shutil.copy2(template_file, tmp_file.name)
                            os.replace(tmp_file.name, dest_file)
                        copied_files.append(dest_file)
                    logger.info(f"Template {'would be' if dry_run else 'was'} copied: {template_file.name}")
                except (PermissionError, IOError) as e:
                    logger.error(f"Error copying {template_file.name}: {str(e)}")
                    raise

    except Exception as e:
        logger.error(f"An error occurred during the template creation process: {str(e)}")
        if not dry_run:
            # Clean up partially copied files
            for file in copied_files:
                file.unlink(missing_ok=True)
        raise

    logger.info("Template creation process completed.")

```

## File: code2prompt/utils/generate_markdown_content.py

- Extension: .py
- Language: python
- Size: 1295 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
def generate_markdown_content(files_data, no_codeblock):
    """
    Generates a Markdown content string from the provided files data.

    Parameters:
    - files_data (list of dict): A list of dictionaries containing file information and content.
    - no_codeblock (bool): Flag indicating whether to disable wrapping code inside markdown code blocks.

    Returns:
    - str: A Markdown-formatted string containing the table of contents and the file contents.
    """
    table_of_contents = [f"- {file['path']}\n" for file in files_data]
    
    content = []
    for file in files_data:
        file_info = (
            f"## File: {file['path']}\n\n"
            f"- Extension: {file['extension']}\n"
            f"- Language: {file['language']}\n"
            f"- Size: {file['size']} bytes\n"
            f"- Created: {file['created']}\n"
            f"- Modified: {file['modified']}\n\n"
        )
        
        if no_codeblock:
            file_code = f"### Code\n\n{file['content']}\n\n"
        else:
            file_code = f"### Code\n\n```{file['language']}\n{file['content']}\n```\n\n"
        
        content.append(file_info + file_code)
    
    return (
        "# Table of Contents\n"
        + "".join(table_of_contents)
        + "\n"
        + "".join(content)
    )

```

## File: code2prompt/utils/include_loader.py

- Extension: .py
- Language: python
- Size: 2709 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import os
from typing import List, Tuple, Callable
from jinja2 import BaseLoader, TemplateNotFound
import threading
from contextlib import contextmanager
import jinja2  # Import jinja2 to resolve the undefined name error


class CircularIncludeError(Exception):
    """Exception raised when a circular include is detected in templates."""

    pass


class IncludeLoader(BaseLoader):
    """
    A custom Jinja2 loader that supports file inclusion with circular dependency detection.

    This loader keeps track of the include stack for each thread to prevent circular includes.
    It raises a CircularIncludeError if a circular include is detected.

    Attributes:
        path (str): The base path for template files.
        encoding (str): The encoding to use when reading template files.
        include_stack (threading.local): Thread-local storage for the include stack.
    """

    def __init__(self, path: str, encoding: str = "utf-8"):
        """
        Initialize the IncludeLoader.

        Args:
            path (str): The base path for template files.
            encoding (str, optional): The encoding to use when reading template files. Defaults to 'utf-8'.
        """
        self.path: str = path
        self.encoding: str = encoding
        self.include_stack: threading.local = threading.local()

    @contextmanager
    def _include_stack_context(self, path):
        if not hasattr(self.include_stack, "stack"):
            self.include_stack.stack = set()
        if path in self.include_stack.stack:
            raise CircularIncludeError(f"Circular include detected: {path}")
        self.include_stack.stack.add(path)
        try:
            yield
        finally:
            self.include_stack.stack.remove(path)

    def get_source(
        self, environment: "jinja2.Environment", template: str
    ) -> Tuple[str, str, Callable[[], bool]]:
        path: str = os.path.join(self.path, template)
        if not os.path.exists(path):
            raise TemplateNotFound(f"{template} (searched in {self.path})")

        with self._include_stack_context(path):
            try:
                with open(path, "r", encoding=self.encoding) as f:
                    source: str = f.read()
            except IOError as e:
                raise TemplateNotFound(
                    template, message=f"Error reading template file: {e}"
                ) from e

        return source, path, lambda: True

    def list_templates(self) -> List[str]:
        """
        List all available templates.

        This method is not implemented for this loader and always returns an empty list.

        Returns:
            List[str]: An empty list.
        """
        return []

```

## File: code2prompt/utils/is_filtered.py

- Extension: .py
- Language: python
- Size: 2378 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains utility functions for filtering files based on include and exclude patterns.
"""

from pathlib import Path
from fnmatch import fnmatch


def is_filtered(
    file_path: Path,
    include_pattern: str = "",
    exclude_pattern: str = "",
    case_sensitive: bool = False,
) -> bool:
    """
    Determine if a file should be filtered based on include and exclude patterns.

    Parameters:
    - file_path (Path): Path to the file to check
    - include_pattern (str): Comma-separated list of patterns to include files
    - exclude_pattern (str): Comma-separated list of patterns to exclude files
    - case_sensitive (bool): Whether to perform case-sensitive pattern matching

    Returns:
    - bool: True if the file should be included, False if it should be filtered out
    """

    def match_pattern(path: str, pattern: str) -> bool:
        if "**" in pattern:
            parts = pattern.split("**")
            return any(fnmatch(path, f"*{p}*") for p in parts if p)
        return fnmatch(path, pattern)

    def match_patterns(path: str, patterns: list) -> bool:
        return any(match_pattern(path, pattern) for pattern in patterns)

    # Convert file_path to string
    file_path_str = str(file_path)

    # Handle case sensitivity
    if not case_sensitive:
        file_path_str = file_path_str.lower()

    # Prepare patterns
    def prepare_patterns(pattern):
        if isinstance(pattern, str):
            return [p.strip().lower() for p in pattern.split(",") if p.strip()]
        elif isinstance(pattern, (list, tuple)):
            return [str(p).strip().lower() for p in pattern if str(p).strip()]
        else:
            return []

    include_patterns = prepare_patterns(include_pattern)
    exclude_patterns = prepare_patterns(exclude_pattern)

    # If no patterns are specified, include the file
    if not include_patterns and not exclude_patterns:
        return True

    # Check exclude patterns first (they take precedence)
    if match_patterns(file_path_str, exclude_patterns):
        return False  # Exclude dotfiles and other specified patterns

    # If include patterns are specified, the file must match at least one
    if include_patterns:
        return match_patterns(file_path_str, include_patterns)

    # If we reach here, there were no include patterns and the file wasn't excluded
    return True

```

## File: code2prompt/utils/price_calculator.py

- Extension: .py
- Language: python
- Size: 4827 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import json
from typing import List, Optional
from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel, ConfigDict, field_validator


class PriceModel(BaseModel):
    price: Optional[float] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    name: str

    @field_validator("price", "input_price", "output_price")
    @classmethod
    def check_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Price must be non-negative")
        return v


class Provider(BaseModel):
    name: str
    models: List[PriceModel]


class TokenPrices(BaseModel):
    providers: List[Provider]


class PriceResult(BaseModel):
    provider_name: str
    model_name: str
    price_input: float
    price_output: float
    total_tokens: int
    total_price: float
    
    model_config = ConfigDict(protected_namespaces=())




@lru_cache(maxsize=1)
def load_token_prices() -> TokenPrices:
    """
    Load token prices from a JSON file.

    Returns:
        TokenPrices: A Pydantic model containing token prices.

    Raises:
        RuntimeError: If there is an error loading the token prices.
    """
    price_file = Path(__file__).parent.parent / "data" / "token_price.json"
    try:
        with price_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return TokenPrices.model_validate(data)
    except (IOError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading token prices: {str(e)}") from e


def calculate_price(token_count: int, price_per_1000: float) -> float:
    """
    Calculates the price based on the token count and price per 1000 tokens.

    Args:
        token_count (int): The total number of tokens.
        price_per_1000 (float): The price per 1000 tokens.

    Returns:
        float: The calculated price.
    """
    return (token_count / 1_000) * price_per_1000


def calculate_prices(
    token_prices: TokenPrices,
    input_tokens: int,
    output_tokens: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[PriceResult]:
    """
    Calculate the prices for a given number of input and output tokens based on token prices.

    Args:
        token_prices (TokenPrices): A Pydantic model containing token prices for different providers and models.
        input_tokens (int): The number of input tokens.
        output_tokens (int): The number of output tokens.
        provider (str, optional): The name of the provider. If specified, only prices for the specified provider will be calculated.
        model (str, optional): The name of the model. If specified, only prices for the specified model will be calculated.

    Returns:
        List[PriceResult]: A list of PriceResult objects containing the calculation results.
    """
    results = []
    total_tokens = input_tokens + output_tokens

    for provider_data in token_prices.providers:
        if provider and provider_data.name.lower() != provider.lower():
            continue

        for model_data in provider_data.models:
            if model and model_data.name.lower() != model.lower():
                continue

            if model_data.price is not None:
                price_input = model_data.price
                price_output = model_data.price
                total_price = calculate_price(total_tokens, model_data.price)
            elif (
                model_data.input_price is not None
                and model_data.output_price is not None
            ):
                price_input = model_data.input_price
                price_output = model_data.output_price
                total_price = calculate_price(
                    input_tokens, price_input
                ) + calculate_price(output_tokens, price_output)
            else:
                continue

            results.append(
                PriceResult(
                    provider_name=provider_data.name,
                    model_name=model_data.name,
                    price_input=price_input,
                    price_output=price_output,
                    total_tokens=total_tokens,
                    total_price=total_price,
                )
            )

    return results


if __name__ == "__main__":
    # Example usage
    token_prices = load_token_prices()
    results = calculate_prices(token_prices, input_tokens=100, output_tokens=50)
    for result in results:
        print(f"Provider: {result.provider_name}")
        print(f"Model: {result.model_name}")
        print(f"Input Price: ${result.price_input:.10f}")
        print(f"Output Price: ${result.price_output:.10f}")
        print(f"Total Tokens: {result.total_tokens}")
        print(f"Total Price: ${result.total_price:.10f}")
        print("---")
```

## File: code2prompt/utils/file_utils.py

- Extension: .py
- Language: python
- Size: 4319 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/utils/file_utils.py

from pathlib import Path
from typing import List, Dict, Any
import logging

from code2prompt.config import Configuration
from code2prompt.utils.is_binary import is_binary
from code2prompt.utils.is_filtered import is_filtered
from code2prompt.utils.is_ignored import is_ignored
from code2prompt.utils.get_gitignore_patterns import get_gitignore_patterns
from code2prompt.core.process_file import process_file

logger = logging.getLogger(__name__)

def process_files(config: Configuration) -> List[Dict[str, Any]]:
    """
    Process files based on the provided configuration.

    Args:
        config (Configuration): Configuration object containing processing options.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing processed file data.
    """
    files_data = []
    for path in config.path:
        path = Path(path)
        gitignore_patterns = get_gitignore_patterns(
            path.parent if path.is_file() else path,
            config.gitignore
        )
        
        if path.is_file():
            file_data = process_single_file(path, gitignore_patterns, config)
            if file_data:
                files_data.append(file_data)
        else:
            files_data.extend(process_directory(path, gitignore_patterns, config))
    
    return files_data

def process_single_file(
    file_path: Path,
    gitignore_patterns: List[str],
    config: Configuration
) -> Dict[str, Any]:
    """
    Process a single file if it meets the criteria.

    Args:
        file_path (Path): Path to the file to process.
        gitignore_patterns (List[str]): List of gitignore patterns.
        config (Configuration): Configuration object containing processing options.

    Returns:
        Dict[str, Any]: Processed file data if the file should be processed, None otherwise.
    """
    if should_process_file(file_path, gitignore_patterns, file_path.parent, config):
        return process_file(
            file_path,
            config.suppress_comments,
            config.line_number,
            config.no_codeblock
        )
    return None

def process_directory(
    directory_path: Path,
    gitignore_patterns: List[str],
    config: Configuration
) -> List[Dict[str, Any]]:
    """
    Process all files in a directory that meet the criteria.

    Args:
        directory_path (Path): Path to the directory to process.
        gitignore_patterns (List[str]): List of gitignore patterns.
        config (Configuration): Configuration object containing processing options.

    Returns:
        List[Dict[str, Any]]: List of processed file data for files that meet the criteria.
    """
    files_data = []
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_data = process_single_file(file_path, gitignore_patterns, config)
            if file_data:
                files_data.append(file_data)
    return files_data

def should_process_file(
    file_path: Path,
    gitignore_patterns: List[str],
    root_path: Path,
    config: Configuration
) -> bool:
    """
    Determine whether a file should be processed based on several criteria.

    Args:
        file_path (Path): Path to the file to check.
        gitignore_patterns (List[str]): List of gitignore patterns.
        root_path (Path): Root path for relative path calculations.
        config (Configuration): Configuration object containing processing options.

    Returns:
        bool: True if the file should be processed, False otherwise.
    """
    logger.debug(f"Checking if should process file: {file_path}")

    if not file_path.is_file():
        logger.debug(f"Skipping {file_path}: Not a file.")
        return False

    if is_ignored(file_path, gitignore_patterns, root_path):
        logger.debug(f"Skipping {file_path}: File is ignored based on gitignore patterns.")
        return False

    if not is_filtered(
        file_path,
        config.filter,
        config.exclude,
        config.case_sensitive
    ):
        logger.debug(f"Skipping {file_path}: File does not meet filter criteria.")
        return False

    if is_binary(file_path):
        logger.debug(f"Skipping {file_path}: File is binary.")
        return False

    logger.debug(f"Processing file: {file_path}")
    return True
```

## File: code2prompt/utils/get_gitignore_patterns.py

- Extension: .py
- Language: python
- Size: 1040 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from code2prompt.utils.parse_gitignore import parse_gitignore
from pathlib import Path

def get_gitignore_patterns(path, gitignore):
    """
    Retrieve gitignore patterns from a specified path or a default .gitignore file.

    This function reads the .gitignore file located at the specified path or uses
    the default .gitignore file in the project root if no specific path is provided.
    It then parses the file to extract ignore patterns and adds a default pattern
    to ignore the .git directory itself.

    Args:
    path (Path): The root path of the project where the default .gitignore file is located.
    gitignore (Optional[str]): An optional path to a specific .gitignore file to use instead of the default.

    Returns:
    Set[str]: A set of gitignore patterns extracted from the .gitignore file.
    """
    if gitignore:
        gitignore_path = Path(gitignore)
    else:
        gitignore_path = Path(path) / ".gitignore"

    patterns = parse_gitignore(gitignore_path)
    patterns.add(".git")
    return patterns
```

## File: code2prompt/utils/add_line_numbers.py

- Extension: .py
- Language: python
- Size: 477 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
def add_line_numbers(code: str) -> str:
    """
    Adds line numbers to each line of the given code.

    Args:
        code (str): The code to add line numbers to.

    Returns:
        str: The code with line numbers added.
    """
    lines = code.splitlines()
    max_line_number = len(lines)
    line_number_width = len(str(max_line_number))
    numbered_lines = [f"{i+1:{line_number_width}} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)
```

## File: code2prompt/utils/should_process_file.py

- Extension: .py
- Language: python
- Size: 1787 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""

This module contains the function to determine 
if a file should be processed based on several criteria.

"""

import logging
from pathlib import Path
from typing import List  # Add this import
from code2prompt.utils.is_binary import is_binary
from code2prompt.utils.is_filtered import is_filtered
from code2prompt.utils.is_ignored import is_ignored

logger = logging.getLogger(__name__)


def should_process_file(
    file_path: Path,
    gitignore_patterns: List[str],  # List is now defined
    root_path: Path,
    filter_patterns: str,  ## comma separated list of patterns
    exclude_patterns: str,  ## comma separated list of patterns
    case_sensitive: bool,
) -> bool:
    """
    Determine whether a file should be processed based on several criteria.
    """
    logger.debug(
        "Checking if should process file: %s", file_path
    )  # Use lazy % formatting

    if not file_path.is_file():
        logger.debug("Skipping %s: Not a file.", file_path)  # Use lazy % formatting
        return False

    if is_ignored(file_path, gitignore_patterns, root_path):
        logger.debug(
            "Skipping %s: File is ignored based on gitignore patterns.", file_path
        )
        return False

    if not is_filtered(
        file_path=file_path,
        include_pattern=filter_patterns,
        exclude_pattern=exclude_patterns,
        case_sensitive=case_sensitive,
    ):
        logger.debug(
            "Skipping %s: File does not meet filter criteria.", file_path
        )  # Use lazy % formatting
        return False

    if is_binary(file_path):
        logger.debug("Skipping %s: File is binary.", file_path)  # Use lazy % formatting
        return False

    logger.debug("Processing file: %s", file_path)  # Use lazy % formatting
    return True

```

## File: code2prompt/utils/display_price_table.py

- Extension: .py
- Language: python
- Size: 3687 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from code2prompt.utils.price_calculator import calculate_prices, load_token_prices

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def format_price(price: float, is_total: bool = False) -> str:
    """
    Formats the given price as a string.

    Args:
        price (float): The price to be formatted.
        is_total (bool, optional): Indicates whether the price is a total. Defaults to False.

    Returns:
        str: The formatted price as a string.

    """
    if is_total:
        return f"${price:.6f}"
    return f"${price /1_000 * 1_000_000 :.2f}"


def format_specific_price(price: float, tokens: int) -> str:
    """
    Formats the specific price based on the given price and tokens.

    Args:
        price (float): The price value.
        tokens (int): The number of tokens.

    Returns:
        str: The formatted specific price.

    """
    return f"${(price * tokens / 1_000):.6f}"


def display_price_table(
    output_tokens: int, provider: str, model: str, token_count: int
):
    """
    Displays a price table with estimated token prices based on the token count and provider's pricing model.

    Args:
        output_tokens (int): The number of output tokens.
        provider (str): The name of the provider.
        model (str): The name of the model.
        token_count (int): The number of input tokens.

    Returns:
        None
    """
    token_prices = load_token_prices()
    if not token_prices:
        return
    price_results = calculate_prices(
        token_prices, token_count, output_tokens, provider, model
    )

    if not price_results:
        click.echo("Error: No matching provider or model found")
        return

    console = Console()

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Input Price\n($/1M tokens)", justify="right", style="yellow")
    table.add_column("Output Price\n($/1M tokens)", justify="right", style="yellow")
    table.add_column("Tokens\nOut | In", justify="right", style="blue")
    table.add_column("Price $\nOut | In", justify="right", style="magenta")
    table.add_column("Total Cost", justify="right", style="red")

    for result in price_results:
        input_price = format_price(result.price_input)
        output_price = format_price(result.price_output)
        specific_input_price = format_specific_price(result.price_input, token_count)
        specific_output_price = format_specific_price(
            result.price_output, output_tokens
        )
        total_price = format_price(result.total_price, is_total=True)

        table.add_row(
            result.provider_name,
            result.model_name,
            input_price,
            output_price,
            f"{token_count:,} | {output_tokens:,}",
            f"{specific_input_price} | {specific_output_price}",
            total_price,
        )

    title = Text("Estimated Token Prices", style="bold white on blue")
    subtitle = Text("All prices in USD", style="italic")

    panel = Panel(
        table, title=title, subtitle=subtitle, expand=False, border_style="blue"
    )

    console.print("\n")
    console.print(panel)
    console.print(
        "\n📊 Note: Prices are based on the token count and provider's pricing model."
    )
    console.print(
        "💡 Tip: 'Price $ In | Out' shows the cost for the specific input and output tokens."
    )
    console.print(
        "⚠️  This is an estimate based on OpenAI's Tokenizer implementation.\n"
    )

```

## File: code2prompt/utils/language_inference.py

- Extension: .py
- Language: python
- Size: 3377 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import os


def infer_language(filename: str, syntax_map: dict) -> str:
    """
    Infers the programming language of a given file based on its extension.

    Parameters:
    - filename (str): The name of the file including its extension.
    - syntax_map (dict): Custom syntax mappings for language inference.

    Returns:
    - str: The inferred programming language as a lowercase string, e.g., "python".
           Returns "unknown" if the language cannot be determined.
    """
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    # Check user-defined syntax map first
    if extension in syntax_map:
        return syntax_map[extension]

    language_map = {
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cs": "csharp",
        ".php": "php",
        ".go": "go",
        ".rs": "rust",
        ".kt": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
        ".dart": "dart",
        ".py": "python",
        ".rb": "ruby",
        ".pl": "perl",
        ".pm": "perl",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".ps1": "powershell",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".sql": "sql",
        ".m": "matlab",
        ".r": "r",
        ".lua": "lua",
        ".jl": "julia",
        ".f": "fortran",
        ".f90": "fortran",
        ".hs": "haskell",
        ".lhs": "haskell",
        ".ml": "ocaml",
        ".erl": "erlang",
        ".ex": "elixir",
        ".exs": "elixir",
        ".clj": "clojure",
        ".coffee": "coffeescript",
        ".groovy": "groovy",
        ".pas": "pascal",
        ".vb": "visualbasic",
        ".asm": "assembly",
        ".s": "assembly",
        ".lisp": "lisp",
        ".cl": "lisp",
        ".scm": "scheme",
        ".rkt": "racket",
        ".fs": "fsharp",
        ".d": "d",
        ".ada": "ada",
        ".nim": "nim",
        ".cr": "crystal",
        ".v": "verilog",
        ".vhd": "vhdl",
        ".tcl": "tcl",
        ".elm": "elm",
        ".zig": "zig",
        ".raku": "raku",
        ".perl6": "raku",
        ".p6": "raku",
        ".vim": "vimscript",
        ".ps": "postscript",
        ".prolog": "prolog",
        ".cobol": "cobol",
        ".cob": "cobol",
        ".cbl": "cobol",
        ".forth": "forth",
        ".fth": "forth",
        ".abap": "abap",
        ".apex": "apex",
        ".sol": "solidity",
        ".hack": "hack",
        ".sml": "standardml",
        ".purs": "purescript",
        ".idr": "idris",
        ".agda": "agda",
        ".lean": "lean",
        ".wasm": "webassembly",
        ".wat": "webassembly",
        ".j2": "jinja2",
        ".md": "markdown",
        ".tex": "latex",
        ".bib": "bibtex",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
        ".dockerfile": "dockerfile",
        ".docker": "dockerfile",
        '.txt': 'plaintext',
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.log': 'log'
    }

    return language_map.get(extension, "unknown")

```

## File: code2prompt/utils/logging_utils.py

- Extension: .py
- Language: python
- Size: 3985 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import logging
import colorlog
import sys

def setup_logger(level="INFO"):
    """Set up the logger with the specified logging level."""
    logger = colorlog.getLogger()
    logger.setLevel(level)

    # Create console handler
    ch = colorlog.StreamHandler()
    ch.setLevel(level)

    # Create formatter with a more structured format
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger

def log_error(message):
    """Log an error message."""
    logger = logging.getLogger()
    logger.error(message)

def log_output_created(output_path):
    """Log a message indicating that an output file has been created."""
    logger = logging.getLogger()
    logger.info(f"Output file created: {output_path}")

def log_clipboard_copy(success):
    """Log a message indicating whether copying to clipboard was successful."""
    logger = logging.getLogger()
    if success:
        success_message = "\033[92m📋 Content copied to clipboard successfully.\033[0m"
        logger.info(success_message)
        print(success_message, file=sys.stderr)
    else:
        logger.error("Failed to copy content to clipboard.")
        print("Failed to copy content to clipboard.", file=sys.stderr)

def log_token_count(token_count):
    """Log the token count."""
    # Calculate the number of tokens in the input 
    token_count_message = f"\n✨ \033[94mToken count: {token_count}\033[0m\n"  # Added color for better display
    print(token_count_message, file=sys.stderr)

def log_token_prices(prices):
    """Log the estimated token prices."""
    # Remove the unused logger variable
    # logger = logging.getLogger()  # Unused variable
    header = "─────────────────────────────────────────────────── Estimated Token Prices ───────────────────────────────────────────────────"
    print(header)
    print("┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓")
    print("┃             ┃                     ┃   Input Price ┃  Output Price ┃         Tokens ┃               Price $ ┃            ┃")
    print("┃ Provider    ┃ Model               ┃ ($/1M tokens) ┃ ($/1M tokens) ┃       In | Out ┃              In | Out ┃ Total Cost ┃")
    print("┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩")
    for price in prices:
        print(f"│ {price['provider']: <11} │ {price['model']: <19} │ {price['input_price']: >13} │ {price['output_price']: >13} │ {price['tokens_in']: >13} | {price['tokens_out']: >13} │ {price['input_cost']: >12} | {price['output_cost']: >12} │ {price['total_cost']: >12} │")
    print("└─────────────┴─────────────────────┴─────────────────┴─────────────────┴────────────────┴─────────────────────┴────────────┘")

```

## File: code2prompt/utils/parse_gitignore.py

- Extension: .py
- Language: python
- Size: 273 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
def parse_gitignore(gitignore_path):
    if not gitignore_path.exists():
        return set()
    with gitignore_path.open("r", encoding="utf-8") as file:
        patterns = set(line.strip() for line in file if line.strip() and not line.startswith("#"))
    return patterns
```

## File: code2prompt/comment_stripper/strip_comments.py

- Extension: .py
- Language: python
- Size: 1717 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains the function to strip comments from code based on the programming language.
"""

from .c_style import strip_c_style_comments
from .html_style import strip_html_style_comments
from .python_style import strip_python_style_comments
from .shell_style import strip_shell_style_comments
from .sql_style import strip_sql_style_comments
from .matlab_style import strip_matlab_style_comments
from .r_style import strip_r_style_comments


def strip_comments(code: str, language: str) -> str:
    """Strips comments from the given code based on the specified programming language.

    Args:
        code (str): The source code from which comments will be removed.
        language (str): The programming language of the source code.

    Returns:
        str: The code without comments.
    """
    if language in [
        "c",
        "cpp",
        "java",
        "javascript",
        "csharp",
        "php",
        "go",
        "rust",
        "kotlin",
        "swift",
        "scala",
        "dart",
        "typescript",
        "typescriptreact",
        "react",
    ]:
        return strip_c_style_comments(code)
    elif language in ["python", "ruby", "perl"]:
        return strip_python_style_comments(code)
    elif language in ["bash", "powershell", "shell"]:
        return strip_shell_style_comments(code)
    elif language in ["html", "xml"]:
        return strip_html_style_comments(code)
    elif language in ["sql", "plsql", "tsql"]:
        return strip_sql_style_comments(code)
    elif language in ["matlab", "octave"]:
        return strip_matlab_style_comments(code)
    elif language in ["r"]:
        return strip_r_style_comments(code)
    else:
        return code

```

## File: code2prompt/comment_stripper/sql_style.py

- Extension: .py
- Language: python
- Size: 336 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_sql_style_comments(code: str) -> str:
    pattern = re.compile(
        r'--.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(
        pattern,
        lambda match: match.group(0) if match.group(0).startswith(("'", '"')) else "",
        code,
    )

```

## File: code2prompt/comment_stripper/c_style.py

- Extension: .py
- Language: python
- Size: 334 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_c_style_comments(code: str) -> str:
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(
        pattern,
        lambda match: match.group(0) if match.group(0).startswith(("'", '"')) else "",
        code,
    )

```

## File: code2prompt/comment_stripper/python_style.py

- Extension: .py
- Language: python
- Size: 357 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_python_style_comments(code: str) -> str:
    pattern = re.compile(
        r'(?s)#.*?$|\'\'\'.*?\'\'\'|""".*?"""|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.MULTILINE,
    )
    return re.sub(
        pattern,
        lambda match: ("" if match.group(0).startswith(("#", "'''", '"""')) else match.group(0)),
        code,
    )

```

## File: code2prompt/comment_stripper/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python

```

## File: code2prompt/comment_stripper/shell_style.py

- Extension: .py
- Language: python
- Size: 720 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
def strip_shell_style_comments(code: str) -> str:
    lines = code.split("\n")
    new_lines = []
    in_multiline_comment = False
    for line in lines:
        if line.strip().startswith("#!"):  # Preserve shebang lines
            new_lines.append(line)
        elif in_multiline_comment:
            if line.strip().endswith("'"):
                in_multiline_comment = False
        elif line.strip().startswith(": '"):
            in_multiline_comment = True
        elif "#" in line:  # Remove single-line comments
            line = line.split("#", 1)[0]
            if line.strip():
                new_lines.append(line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines).strip()

```

## File: code2prompt/comment_stripper/r_style.py

- Extension: .py
- Language: python
- Size: 323 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_r_style_comments(code: str) -> str:
    pattern = re.compile(
        r'#.*?$|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(
        pattern,
        lambda match: match.group(0) if match.group(0).startswith(("'", '"')) else "",
        code,
    )

```

## File: code2prompt/comment_stripper/matlab_style.py

- Extension: .py
- Language: python
- Size: 328 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_matlab_style_comments(code: str) -> str:
    pattern = re.compile(
        r'%.*?$|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(
        pattern,
        lambda match: match.group(0) if match.group(0).startswith(("'", '"')) else "",
        code,
    )

```

## File: code2prompt/comment_stripper/html_style.py

- Extension: .py
- Language: python
- Size: 148 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
import re

def strip_html_style_comments(code: str) -> str:
    pattern = re.compile(r"<!--.*?-->", re.DOTALL)
    return re.sub(pattern, "", code)

```

## File: code2prompt/templates/create-readme.j2

- Extension: .j2
- Language: jinja2
- Size: 3897 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2
## Role and Expertise:

You are an elite technical writer and documentation specialist with deep expertise in growth hacking, open-source software, and GitHub best practices. Your mission is to craft an exceptional README.md file that will significantly boost a project's visibility, adoption, and community engagement.

## Your Task:

1. Analyze the provided code base meticulously, focusing on:
   - Core functionality and unique selling points
   - Technical architecture and design patterns
   - Integration capabilities and extensibility
   - Performance characteristics and scalability
   - Security features and compliance standards (if applicable)

2. Generate a comprehensive list of key components for an ideal README.md. Present these in a <keycomponents> section, structured as a markdown checklist.

3. Craft a stellar README.md file, presented in an <artifact> section. This README should not only inform but inspire and engage potential users and contributors.

## README.md Requirements:

Your README.md must include:

1. Project Title and Description
   - Concise, compelling project summary
   - Eye-catching logo or banner (placeholder if not provided)

2. Badges
   - Build status, version, license, code coverage, etc.

3. Key Features
   - Bulleted list of main functionalities and unique selling points

4. Quick Start Guide
   - Step-by-step installation instructions
   - Basic usage example

5. Detailed Documentation
   - In-depth usage instructions
   - API reference (if applicable)
   - Configuration options

6. Examples and Use Cases
   - Code snippets demonstrating common scenarios
   - Links to more extensive examples or demos

7. Project Structure
   - Brief overview of the repository's organization

8. Dependencies
   - List of required libraries, frameworks, and tools
   - Compatibility information (OS, language versions, etc.)

9. Contributing Guidelines
   - How to submit issues, feature requests, and pull requests
   - Coding standards and commit message conventions

10. Testing
    - Instructions for running tests
    - Information on the testing framework used

11. Deployment
    - Guidelines for deploying the project (if applicable)

12. Roadmap
    - Future plans and upcoming features

13. License
    - Clear statement of the project's license

14. Acknowledgments
    - Credits to contributors, inspirations, or related projects

15. Contact Information
    - How to reach the maintainers
    - Links to community channels (Slack, Discord, etc.)

## Styling and Formatting:

- Use clear, concise language optimized for skimming and quick comprehension
- Employ a friendly, professional tone that reflects the project's ethos
- Utilize Markdown features effectively:
  - Hierarchical headings (H1 for title, H2 for main sections, H3 for subsections)
  - Code blocks with appropriate language highlighting
  - Tables for structured data
  - Blockquotes for important notes or quotes
  - Horizontal rules to separate major sections
- Include a table of contents for easy navigation
- Use emojis sparingly to add visual interest without overwhelming
- As expert Github expert use all the markdown Github flavor you can to make the README.md more appealing
- Use Github badges that make sense for the project

## Output Format:

Structure your response as follows:

<keycomponents>
  [Checklist of key README components]
</keycomponents>

<artifact>
  [Full content of the README.md]
</artifact>

Remember to tailor the content, tone, and technical depth to the project's target audience, whether they are beginners, experienced developers, or a specific niche within the tech community.

---
## The codebase:

<codebase>

<toc>
## Table of Contents

{% for file in files %}{{ file.path }}
{% endfor %}
</toc>

<code>
{% for file in files %}
## {{ file.path }}

```{{ file.language }}
{{ file.content }}
```

{% endfor %}
</code>

</codebase>

```

## File: code2prompt/templates/default.j2

- Extension: .j2
- Language: jinja2
- Size: 282 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2
# Code summary
{% for file in files %}- {{ file.path }}
{% endfor %}

## Files

{% for file in files %}
## {{ file.path }}

- Language: {{ file.language }}
- Size: {{ file.size }} bytes
- Last modified: {{ file.modified }}

```{{ file.language }}
{{ file.content }}
```
{% endfor %}
```

## File: code2prompt/templates/improve-this-prompt.j2

- Extension: .j2
- Language: jinja2
- Size: 2714 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2
## Who you are
You are an elite prompt engineer with unparalleled expertise in crafting sophisticated and effective prompts. Your task is to significantly enhance the following prompt:

## The Prompt:

<prompt>
{{input:prompt}}
</prompt>
    
### Your Task

1. Analyze the input prompt, identifying its:
    - Primary objective
    - Target audience
    - Key constraints or requirements
    - Potential weaknesses or areas for improvement

2. Apply at least two of the following advanced prompting techniques to enhance the prompt:
    - Chain of Thought (CoT): Break down complex reasoning into steps.
    - Tree of Thought (ToT): Explore multiple reasoning paths.
    - Few-Shot Learning: Provide relevant examples.
    - Role-Playing: Assume a specific persona or expertise.
    - Metacognitive Prompting: Encourage self-reflection.

3. Craft your improved prompt, ensuring it is:
    - Clear and unambiguous
    - Specific and detailed
    - Designed to elicit high-quality, relevant responses
    - Flexible enough to accommodate various scenarios
    - Structured to maximize the AI's capabilities

### Examples of Technique Application

Chain of Thought (CoT):
"To answer this question, let's break it down into steps:
1. First, consider...
2. Next, analyze...
3. Finally, synthesize..."

Role-Playing:
"Imagine you are a renowned expert in [field]. Given your extensive experience, how would you approach..."

### Quality Metrics

Evaluate your improved prompt based on:
1. Clarity: Is the prompt easy to understand?
2. Specificity: Does it provide clear guidelines and expectations?
3. Engagement: Does it inspire creative and thoughtful responses?
4. Versatility: Can it be applied to various scenarios within the context?
5. Depth: Does it encourage detailed and nuanced responses?

### Iterative Refinement

After crafting your initial improved prompt:
1. Critically review it against the quality metrics.
2. Identify at least one area for further improvement.
3. Refine the prompt based on this insight.
4. Repeat this process once more for optimal results.

### Output Format

Present your work in the following structure:

1. Original Prompt Analysis in markdown format, in xml tags <analysis>
2. First version Improved Prompt (in markdown format) in <prompt_v1>
3. Explanation of Applied Techniques in <techniques>
4. Quality Metric Evaluation in <metrics>
5. Iterative Refinement Process in <refinement>
6. Final Thoughts on Improvement in <final_thoughts>
7. The final prompt in markdown format in <prompt_final>

By following this structured approach, you'll create a significantly enhanced prompt that drives high-quality AI outputs and addresses the specific needs of the given context.






```

## File: code2prompt/templates/create-function.j2

- Extension: .j2
- Language: jinja2
- Size: 2566 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2

# Write a function 

## Your Role

You are an elite software developer with extensive. You have a strong background in debugging complex issues and optimizing code performance. 

## Task Overview

You need provide a correct and tested implementation of the `{%input:function_name%}`


## Detailed Requirements

{%input:function_description%}

## Output Format

Please provide your response in the following structured format:

1. <observed>
   Detailed specification of the is_filtered function based on the given information and your analysis. You must include the function signature, parameters, and expected behavior.
   Use markdown.
</observed>

2. <spec_tests>
   Provide a set of test cases that cover different scenarios for the is_filtered function. Include both positive and negative test cases to validate the implementation.
   Propose edge case scenarios that might challenge the function's logic.
   Use markdown code blocks to format the test cases.
</spec_test>


3. <tests>
   <test filename="function_name.py">
    Unit test in markdown format, use code block to format the test
   </test>
    ... other tests ...
</tests>


3. <first_implentation>
   Describe your initial implementation of the  function, including any assumptions or design decisions you made. Explain how you approached the problem and any challenges you encountered.
   Be careful to follow the specification and provide a clear explanation of your code. The function must pass the provided test cases.
   Format as a code block in markdown
</first_implementation>

4. <evaluation_and_critics>
   Evaluate the strengths and weaknesses of your initial implementation. Discuss any limitations or areas for improvement in the code.
</evaluation_and_critics>

5. <artifact>
   Provide the complete, updated code for the  function. Include all existing comments and add new comments where necessary to explain the changes and their purpose.
   Format as a code block in markdown
</artifact


## Additional Guidelines

- Ensure your solution is compatible the current codebase and follows the existing coding style and conventions.
- Provide clear, concise comments in the code to explain complex logic or non-obvious decisions.
- If you make assumptions about the existing codebase structure, clearly state these assumptions.

---
## The codebase:

<codebase>

<toc>
## Table of Contents

{% for file in files %}{{ file.path }}
{% endfor %}
</toc>

<code>
{% for file in files %}
## {{ file.path }}

```{{ file.language }}
{{ file.content }}
```

{% endfor %}
</code>

</codebase>

```

## File: code2prompt/templates/analyze-code.j2

- Extension: .j2
- Language: jinja2
- Size: 4926 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2
# Elite Code Analyzer and Improvement Strategist 2.0

## Role and Goal
You are a world-class software architect and code quality expert with decades of experience across various programming languages, paradigms, and industries. Your mission is to analyze the provided codebase comprehensively, uncover its strengths and weaknesses, and develop a strategic improvement plan that balances immediate gains with long-term sustainability.

## Core Competencies
- Mastery of software architecture principles, design patterns, and best practices across multiple paradigms (OOP, functional, etc.)
- Deep expertise in performance optimization, security hardening, and scalability enhancement for both monolithic and distributed systems
- Proven track record in successful large-scale refactoring and technical debt reduction
- Cutting-edge knowledge of modern development frameworks, cloud technologies, and DevOps practices
- Strong understanding of collaborative development processes and team dynamics

## Task Breakdown

1. Initial Assessment
   - Identify the programming language(s), frameworks, and overall architecture
   - Determine the scale and complexity of the codebase
   - Assess the development environment and team structure

2. Multi-Dimensional Analysis (Utilize Tree of Thought)
   a. Functionality and Business Logic
   b. Architectural Design and Patterns
   c. Code Quality and Maintainability
   d. Performance and Scalability
   e. Security and Data Protection
   f. Testing and Quality Assurance
   g. DevOps and Deployment Processes
   h. Documentation and Knowledge Management

3. Improvement Identification (Apply Chain of Thought)
   For each analyzed dimension:
   - Describe the current state
   - Envision the ideal state
   - Identify the gap between current and ideal
   - Generate potential improvements, considering:
     - Short-term quick wins
     - Medium-term enhancements
     - Long-term strategic changes

4. Holistic Evaluation (Implement Ensemble Prompting)
   Compile and synthesize insights from multiple perspectives:
   - Senior Developer: Code quality and maintainability
   - DevOps Engineer: Scalability and operational efficiency
   - Security Specialist: Vulnerability assessment and risk mitigation
   - Product Manager: Feature delivery and business value
   - End-user: Usability and performance perception

5. Strategic Improvement Plan (Use Step-by-Step Reasoning)
   Develop a comprehensive plan that:
   - Prioritizes improvements based on:
     - Impact on system quality and business value
     - Implementation complexity and risk
     - Resource requirements and availability
     - Interdependencies between improvements
   - Balances quick wins with long-term architectural enhancements
   - Considers team dynamics and skill development needs
   - Incorporates continuous improvement and feedback loops

## Output Format

<keypoints>
[Present as a markdown checklist, categorized by analysis dimensions]
</keypoints>

<artifact>
[Structured improvement plan with clear sections for immediate actions, short-term goals, and long-term vision]
</artifact>

## Additional Instructions

- Tailor your analysis and recommendations to the specific programming languages and paradigms used in the codebase
- Use industry-standard metrics and benchmarks to support your analysis where applicable
- Provide concrete examples or pseudo-code to illustrate complex concepts or proposed changes
- Address potential challenges in implementing improvements, including team resistance or resource constraints
- Suggest collaborative approaches and tools to facilitate the improvement process
- Consider the impact of proposed changes on the entire software development lifecycle

## Logging Level Configuration

To configure the log level in the command line, you can use the `--log-level` option when running the `code2prompt` command. This option allows you to specify the desired logging level, such as DEBUG, INFO, WARNING, ERROR, or CRITICAL.

## Reflection and Continuous Improvement

After completing the analysis and plan:
- Identify areas where the analysis could be deepened with additional tools or information
- Reflect on how the improvement strategy aligns with current industry trends and emerging technologies
- Propose a mechanism for tracking the progress and impact of implemented improvements
- Suggest how this analysis process itself could be enhanced for future iterations

Remember, your goal is to provide a transformative yet pragmatic roadmap that elevates the quality, performance, and maintainability of the codebase while considering the realities of the development team and business constraints.
---
## The codebase:

<codebase>

<toc>
## Table of Contents

{% for file in files %}{{ file.path }}
{% endfor %}
</toc>

<code>
{% for file in files %}
## {{ file.path }}

```{{ file.language }}
{{ file.content }}
```

{% endfor %}
</code>

</codebase>

```

## File: code2prompt/templates/code-review.j2

- Extension: .j2
- Language: jinja2
- Size: 4190 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```jinja2
# Role: Expert Code Reviewer and Software Architect

You are a world-class code reviewer and software architect with decades of experience across multiple programming languages and paradigms. Your expertise spans from low-level optimization to high-level architectural design. You have a keen eye for code quality, security, and scalability.

# Task Overview

Conduct a thorough code review of the provided files, focusing on improving code quality, robustness, simplicity, and documentation. Your review should be insightful, actionable, and prioritized.

# Review Process

1. Initial Assessment:
   - Quickly scan all files to understand the overall structure and purpose of the code.
   - Identify the primary programming paradigms and architectural patterns in use.

2. Detailed Analysis:
   - Examine each file in depth, considering the following aspects:
     a. Code Quality
     b. Robustness and Error Handling
     c. Simplification and Refactoring
     d. Naming and Documentation
     e. Security and Best Practices
     f. Performance and Scalability

3. Prioritization:
   - Categorize your findings into:
     - Critical: Issues that could lead to bugs, security vulnerabilities, or significant performance problems.
     - Important: Violations of best practices or areas for significant improvement.
     - Minor: Style issues or small optimizations.

4. Recommendations:
   - For each issue, provide:
     - A clear explanation of the problem
     - The potential impact or risk
     - A suggested solution or improvement
   - Use the following format for each recommendation:
     ```
     [Category: Critical/Important/Minor]
     Issue: [Brief description]
     Location: [File name and line number(s)]
     Impact: [Potential consequences]
     Recommendation: [Suggested fix or improvement]
     Example:
       Before: [Code snippet or description]
       After: [Improved code snippet or description]
     Rationale: [Explanation of the benefits of this change]
     ```

5. Overall Assessment:
   - Provide a high-level summary of the codebase's strengths and weaknesses.
   - Suggest any architectural or structural changes that could benefit the project.

6. Large Codebases:
   - If reviewing a large codebase or multiple interconnected files, focus on:
     a. Identifying common patterns or anti-patterns across files
     b. Assessing overall architecture and suggesting improvements
     c. Highlighting any inconsistencies in style or approach between different parts of the codebase

7. Testing and Quality Assurance:
   - Evaluate the existing test coverage (if any)
   - Suggest areas where additional unit tests could be beneficial
   - Recommend integration or end-to-end tests if appropriate

8. Self-Reflection:
   - Acknowledge any areas where the analysis might be limited due to lack of context or specific domain knowledge
   - Suggest specific questions or areas where human developer input would be valuable

# Guidelines

- Preserve existing functionality unless explicitly improving error handling or security.
- Infer and respect the original code's intent.
- Focus on impactful improvements rather than nitpicking minor style issues.
- If any part of the original code is unclear, state your assumptions and request clarification.
- Consider the broader context and potential scalability of the code.

# Output Format

0. Initial Assessment and all your reflexions in <initial_assessment>
1. High-Priority Issues (Critical and Important findings) in tags <high_priority>
2. Other Recommendations (Minor issues and general improvements) under <other_recommendations>
3. Architectural Considerations (If applicable) in <architecture>
4. Testing Recommendations in <testing>
5. Limitations and Further Inquiries in <limitations>
6. Conclusion and Next Steps in <conclusion>

Your review should be comprehensive, insightful, and actionable, providing clear value to the development team.

---
## The codebase:

<codebase>

<toc>
## Table of Contents

{% for file in files %}{{ file.path }}
{% endfor %}
</toc>

<code>
{% for file in files %}
## {{ file.path }}

```{{ file.language }}
{{ file.content }}
```

{% endfor %}
</code>

</codebase>

```

## File: code2prompt/commands/interactive_selector.py

- Extension: .py
- Language: python
- Size: 12507 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
from typing import List, Dict, Set, Tuple
import os
from pathlib import Path
from prompt_toolkit import Application
from prompt_toolkit.layout.containers import VSplit, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.scrollable_pane import ScrollablePane
from prompt_toolkit.widgets import Frame
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
import signal

# Constant for terminal height adjustment
TERMINAL_HEIGHT_ADJUSTMENT = 3


class InteractiveFileSelector:
    """Interactive file selector."""

    def __init__(self, paths: List[Path], selected_files: List[Path]):
        self.paths: List[Path] = paths.copy()
        self.start_line: int = 0
        self.cursor_position: int = 0
        self.formatted_tree: List[str] = []
        self.tree_paths: List[Path] = []
        self.tree_full_paths: List[str] = []
        self.kb = self._create_key_bindings()
        self.selected_files: Set[str] = set(
            [str(Path(file).resolve()) for file in selected_files]
        )
        self.selection_state: Dict[str, Set[str]] = {}  # State tracking for selections
        self.app = self._create_application(self.kb)

    def _get_terminal_height(self) -> int:
        """Get the height of the terminal."""
        return os.get_terminal_size().lines

    def _get_directory_tree(self) -> Dict[Path, Dict]:
        """Get a combined directory tree for the given paths."""
        tree: Dict[Path, Dict] = {}
        for path in self.paths:
            current = tree  # Start from the root of the tree
            for part in Path(path).parts:
                if part not in current:  # Check if part is already in the current level
                    current[part] = {}  # Create a new dictionary for the part
                current = current[part]  # Move to the next level in the tree
        return tree

    def _format_tree(
        self, tree: Dict[Path, Dict], indent: str = "", parent_dir: str = ""
    ) -> Tuple[List[str], List[Path], List[str]]:
        """Format the directory tree into a list of strings."""
        lines: List[str] = []
        tree_paths: List[Path] = []
        tree_full_paths: List[str] = []
        for i, (file_path, subtree) in enumerate(tree.items()):
            is_last = i == len(tree) - 1
            prefix = "└── " if is_last else "├── "
            line = f"{indent}{prefix}{Path(file_path).name}"
            lines.append(line)
            resolved_path = Path(parent_dir, file_path).resolve()
            tree_paths.append(resolved_path)
            tree_full_paths.append(
                str(resolved_path)
            )  # Store the full path as a string
            if subtree:
                extension = " " if is_last else "│ "
                sub_lines, sub_tree_paths, sub_full_paths = self._format_tree(
                    subtree, indent + extension, str(resolved_path)
                )
                lines.extend(sub_lines)
                tree_paths.extend(sub_tree_paths)
                tree_full_paths.extend(
                    sub_full_paths
                )  # Merge the full paths from the subtree
        return lines, tree_paths, tree_full_paths

    def _validate_cursor_position(self) -> None:
        """Ensure cursor position is valid."""
        if self.cursor_position < 0:
            self.cursor_position = 0
        elif self.cursor_position >= len(self.formatted_tree):
            self.cursor_position = len(self.formatted_tree) - 1

    def _get_visible_lines(self) -> int:
        """Calculate the number of visible lines based on terminal height."""
        terminal_height = self._get_terminal_height()
        return terminal_height - TERMINAL_HEIGHT_ADJUSTMENT  # Use constant

    def _get_formatted_text(self) -> List[tuple]:
        """Generate formatted text for display."""
        result = []
        # Ensure that formatted_tree and tree_paths have the same length
        if len(self.formatted_tree) == len(self.tree_paths):
            visible_lines = self._get_visible_lines()
            # Calculate the end line for the loop
            end_line = min(self.start_line + visible_lines, len(self.formatted_tree))
            for i in range(self.start_line, end_line):
                line = self.formatted_tree[i]
                style = "class:cursor" if i == self.cursor_position else ""
                # Ensure cursor_position is valid
                self._validate_cursor_position()
                # Get the full path
                file_path = str(self.tree_full_paths[i])
                is_dir = os.path.isdir(file_path)
                # Check if the full path is selected
                is_selected = file_path in self.selected_files
                # Update checkbox based on selection state
                checkbox = "[X]" if is_selected else "   " if is_dir else "[ ]"
                if file_path in self.selection_state:
                    if len(self.selection_state[file_path]) == len(self.tree_paths):
                        checkbox = "[X]"
                # Append formatted line to result
                result.append((style, f"{checkbox} {line}\n"))
        return result

    def _toggle_file_selection(self, current_item: str) -> None:
        """Toggle the selection of the current item."""
        # Convert current_item to string to use with startswith
        current_item_str = str(current_item)
        if current_item_str in self.selected_files:
            self.selected_files.remove(current_item_str)
            # Unselect all descendants
            if current_item_str in self.selection_state:
                for descendant in self.selection_state[current_item_str]:
                    self.selected_files.discard(descendant)
                del self.selection_state[current_item_str]
        else:
            self.selected_files.add(current_item_str)
            # Select all descendants
            self.selection_state[current_item_str] = {
                descendant
                for descendant in self.tree_paths
                if str(descendant).startswith(current_item_str)
            }

    def _get_current_item(self) -> str:
        """Get the current item based on cursor position."""
        if 0 <= self.cursor_position < len(self.tree_paths):
            current_item = self.tree_full_paths[self.cursor_position]
            return current_item  # Return the full path
        return None  # Return None if no valid path is found

    def _resize_handler(self, _event) -> None:
        """Handle terminal resize event."""
        self.start_line = max(0, self.cursor_position - self._get_visible_lines() + 1)
        self.app.invalidate()  # Invalidate the application to refresh the layout

    def run(self) -> List[Path]:
        """Run the interactive file selection."""
        self._check_paths()
        tree = self._get_directory_tree()
        self.formatted_tree, self.tree_paths, self.tree_full_paths = self._format_tree(
            tree
        )
        signal.signal(signal.SIGWINCH, self._resize_handler)
        self.app.run()
        list_selected_files : List[Path] = []
        for f in self.selected_files:
            list_selected_files.append(Path(f))
        print(list_selected_files)
        return list_selected_files

    def _create_key_bindings(self) -> KeyBindings:
        """Create and return key bindings for the application."""
        kb = KeyBindings()

        @kb.add("q")
        def quit_application(event):
            event.app.exit()

        @kb.add("up")
        def move_cursor_up(_event):
            if self.cursor_position > 0:
                self.cursor_position -= 1
                # Update start_line if needed for scrolling
                if self.cursor_position < self.start_line:
                    self.start_line = self.cursor_position
                self._validate_cursor_position()  # Validate after moving
                self.app.invalidate()  # Refresh the display after moving

        @kb.add("down")
        def move_cursor_down(_event):
            if self.cursor_position < len(self.formatted_tree) - 1:
                self.cursor_position += 1
                # Update start_line if needed for scrolling
                if self.cursor_position >= self.start_line + self._get_visible_lines():
                    self.start_line += 1
                self._validate_cursor_position()  # Validate after moving
                self.app.invalidate()  # Refresh the display after moving

        @kb.add("pageup")
        def page_up(_event):
            self.cursor_position = max(
                0, self.cursor_position - self._get_visible_lines()
            )
            if self.cursor_position < self.start_line:
                self.start_line = (
                    self.cursor_position
                )  # Adjust start_line to keep the cursor in view
            self.app.invalidate()  # Refresh the display after moving

        @kb.add("pagedown")
        def page_down(_event):
            self.cursor_position = min(
                len(self.formatted_tree) - 1,
                self.cursor_position + self._get_visible_lines(),
            )
            if self.cursor_position >= self.start_line + self._get_visible_lines():
                self.start_line = (
                    self.cursor_position - self._get_visible_lines() + 1
                )  # Adjust start_line to keep the cursor in view
            self.app.invalidate()  # Refresh the display after moving

        @kb.add("space")
        def toggle_selection(_event):
            current_item = self._get_current_item()  # Get the current item as a Path
            if current_item:  # Ensure current_item is not None
                self._toggle_file_selection(
                    current_item
                )  # Pass the Path object directly
                self.app.invalidate()  # Refresh the display after toggling

        @kb.add("enter")
        def confirm_selection(_event):
            self.app.exit()

        return kb

    def _get_selected_files_text(self) -> str:
        """Get the selected files text."""
        if self.selected_files:
            return f"Selected: {len(self.selected_files)} file(s)"
        return "Selected: 0 file(s): None"

    def _create_application(self, kb) -> Application:
        """Create and return the application instance."""
        tree_window = Window(
            content=FormattedTextControl(self._get_formatted_text, focusable=True),
            width=60,
            dont_extend_width=True,
            wrap_lines=False,
        )
        scrollable_tree = ScrollablePane(tree_window)
        instructions = (
            "Instructions:\n"
            "-------------\n"
            "1. Use ↑ and ↓ to navigate\n"
            "2. Press Space to select/deselect an item\n"
            "3. Press Enter to confirm your selection\n"
            "4. Press q to quit the selection process\n"
        )
        layout = Layout(
            VSplit(
                [
                    Frame(scrollable_tree, title="File Tree"),
                    Window(width=1, char="│"),
                    HSplit(
                        [
                            Window(
                                content=FormattedTextControl(instructions), height=5
                            ),
                            Window(height=1),
                            Window(
                                content=FormattedTextControl(
                                    self._get_selected_files_text
                                ),
                                height=10,
                            ),
                        ],
                    ),
                ],
                padding=1,
            )
        )
        style = Style.from_dict(
            {
                "cursor": "bg:#00ff00 #000000",
                "frame.border": "#888888",
            }
        )

        return Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            style=style,
            mouse_support=True,
        )

    def _check_paths(self) -> None:
        """Check if the provided paths are valid."""
        if not self.paths or any(not path for path in self.paths):
            raise ValueError(
                "A valid list of paths must be provided for interactive mode."
            )

```

## File: code2prompt/commands/generate.py

- Extension: .py
- Language: python
- Size: 2795 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
"""
This module contains the GenerateCommand class, which is responsible for generating
markdown content from code files based on the provided configuration.
"""

from typing import List, Dict, Any
from code2prompt.core.process_files import process_files
from code2prompt.core.generate_content import generate_content
from code2prompt.core.write_output import write_output
from code2prompt.utils.count_tokens import count_tokens
from code2prompt.utils.logging_utils import log_token_count
from code2prompt.utils.display_price_table import display_price_table
from code2prompt.commands.base_command import BaseCommand


class GenerateCommand(BaseCommand):
    """Command for generating markdown content from code files."""

    def execute(self) -> None:
        """Execute the generate command."""
        self.logger.info("Generating markdown...")
        file_paths = self._process_files(syntax_map=self.config.syntax_map)  # Pass syntax_map here
        content = self._generate_content(file_paths)
        self._write_output(content)

        if self.config.price:
            self.display_token_count_and_price(content)
        elif self.config.tokens:
            self.display_token_count(content)

        self.logger.info("Generation complete.")

    def _process_files(self, syntax_map: dict) -> List[Dict[str, Any]]:
        """Process files based on the configuration."""
        all_files_data = []
        files_data = process_files(
            file_paths=self.config.path,
            line_number=self.config.line_number,
            no_codeblock=self.config.no_codeblock,
            suppress_comments=self.config.suppress_comments,
            syntax_map=syntax_map,  # Pass syntax_map here
        )
        all_files_data.extend(files_data)
        return all_files_data

    def _generate_content(self, files_data: List[Dict[str, Any]]) -> str:
        """Generate content from processed files data."""
        return generate_content(files_data, self.config.dict())

    def _write_output(self, content: str) -> None:
        """Write the generated content to output."""
        write_output(content, self.config.output, copy_to_clipboard=True)

    def display_token_count_and_price(self, content: str) -> None:
        """Handle token counting and price calculation if enabled."""
        token_count = count_tokens(content, self.config.encoding)
        model = self.config.model
        provider = self.config.provider
        display_price_table(token_count, provider, model, self.config.output_tokens)
        log_token_count(token_count)

        
    def display_token_count(self, content: str) -> None:
        """Display the token count if enabled."""
        token_count = count_tokens(content, self.config.encoding)
        log_token_count(token_count)
        

```

## File: code2prompt/commands/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python

```

## File: code2prompt/commands/base_command.py

- Extension: .py
- Language: python
- Size: 2477 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/commands/base_command.py

from abc import ABC, abstractmethod
import logging
from code2prompt.config import Configuration

class BaseCommand(ABC):
    """
    Abstract base class for all commands in the code2prompt tool.

    This class defines the basic structure and common functionality
    for all command classes. It ensures that each command has access
    to the configuration and a logger, and defines an abstract execute
    method that must be implemented by all subclasses.

    Attributes:
        config (Configuration): The configuration object for the command.
        logger (logging.Logger): The logger instance for the command.
    """

    def __init__(self, config: Configuration, logger: logging.Logger):
        """
        Initialize the BaseCommand with configuration and logger.

        Args:
            config (Configuration): The configuration object for the command.
            logger (logging.Logger): The logger instance for the command.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command.

        This method must be implemented by all subclasses to define
        the specific behavior of each command.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def log_start(self) -> None:
        """
        Log the start of the command execution.
        """
        self.logger.info(f"Starting execution of {self.__class__.__name__}")

    def log_end(self) -> None:
        """
        Log the end of the command execution.
        """
        self.logger.info(f"Finished execution of {self.__class__.__name__}")

    def handle_error(self, error: Exception) -> None:
        """
        Handle and log any errors that occur during command execution.

        Args:
            error (Exception): The exception that was raised.
        """
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}", exc_info=True)

    def validate_config(self) -> bool:
        """
        Validate the configuration for the command.

        This method should be overridden by subclasses to perform
        command-specific configuration validation.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        return True
```

## File: code2prompt/commands/analyze.py

- Extension: .py
- Language: python
- Size: 2140 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```python
# code2prompt/commands/analyze.py

from pathlib import Path
from typing import Dict

from code2prompt.commands.base_command import BaseCommand
from code2prompt.utils.analyzer import (
    analyze_codebase,
    format_flat_output,
    format_tree_output,
    get_extension_list,
)


class AnalyzeCommand(BaseCommand):
    """Command for analyzing the codebase structure."""

    def execute(self) -> None:
        """Execute the analyze command."""
        self.logger.info("Analyzing codebase...")

        for path in self.config.path:
            self._analyze_path(Path(path))

        self.logger.info("Analysis complete.")

    def _analyze_path(self, path: Path) -> None:
        """
        Analyze a single path and output the results.

        Args:
            path (Path): The path to analyze.
        """
        extension_counts, extension_dirs = analyze_codebase(path)

        if not extension_counts:
            self.logger.warning(f"No files found in {path}")
            return

        if self.config.format == "flat":
            output = format_flat_output(extension_counts)
        else:
            output = format_tree_output(extension_dirs)

        print(output)

        print("\nComma-separated list of extensions:")
        print(get_extension_list(extension_counts))

        if self.config.tokens:
            total_tokens = self._count_tokens(extension_counts)
            self.logger.info(f"Total tokens in codebase: {total_tokens}")

    def _count_tokens(self, extension_counts: Dict[str, int]) -> int:
        """
        Count the total number of tokens in the codebase.

        Args:
            extension_counts (Dict[str, int]): A dictionary of file extensions and their counts.

        Returns:
            int: The total number of tokens.
        """
        total_tokens = 0
        for _ext, count in extension_counts.items():
            # This is a simplified token count. You might want to implement a more
            # sophisticated counting method based on the file type.
            total_tokens += count * 100  # Assuming an average of 100 tokens per file

        return total_tokens

```

## File: code2prompt/data/token_price.json

- Extension: .json
- Language: json
- Size: 5962 bytes
- Created: 2024-11-03 11:39:46
- Modified: 2024-11-03 11:39:46

### Code

```json
{
    "description": "This file contains the price of tokens for different models. Prices are in USD for 1000 tokens.",
    "providers": [
      {
        "name": "OpenAI",
        "models": [
          {
            "name": "GPT-4o",
            "input_price": 0.005,
            "output_price": 0.015
          },
          {
            "name": "GPT4o-mini",
            "input_price": 0.000015,
            "output_price": 0.00006
          },
          {
            "name": "GPT-4 (8K)",
            "input_price": 0.03,
            "output_price": 0.06
          },
          {
            "name": "GPT-4 Turbo",
            "input_price": 0.01,
            "output_price": 0.03
          },
          {
            "name": "GPT-3.5-turbo",
            "input_price": 0.0005,
            "output_price": 0.0015
          }
        ]
      },
      {
        "name": "Anthropic",
        "models": [
          {
            "name": "Claude 3 (Opus)",
            "input_price": 0.015,
            "output_price": 0.075
          },
          {
            "name": "Claude 3.5 (Sonnet)",
            "input_price": 0.003,
            "output_price": 0.015
          },
          {
            "name": "Claude 3 (Haiku)",
            "input_price": 0.00025,
            "output_price": 0.00125
          }
        ]
      },
      {
        "name": "Google",
        "models": [
          {
            "name": "Gemini 1.5 Pro",
            "input_price": 0.0035,
            "output_price": 0.007
          },
          {
            "name": "Gemini 1.5 Flash",
            "input_price": 0.00035,
            "output_price": 0.0007
          }
        ]
      },
      {
        "name": "Groq",
        "models": [
          {
            "name": "Llama 3 70b",
            "input_price": 0.00059,
            "output_price": 0.00079
          },
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.00024,
            "output_price": 0.00024
          }
        ]
      },
      {
        "name": "Replicate",
        "models": [
          {
            "name": "Llama 3 70b",
            "input_price": 0.00065,
            "output_price": 0.00275
          },
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0003,
            "output_price": 0.001
          }
        ]
      },
      {
        "name": "Mistral",
        "models": [
          {
            "name": "mistral-large-2402",
            "input_price": 0.004,
            "output_price": 0.012
          },
          {
            "name": "codestral-2405",
            "input_price": 0.001,
            "output_price": 0.003
          },
          {
            "name": "Mixtral 8x22B",
            "input_price": 0.002,
            "output_price": 0.006
          },
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0007,
            "output_price": 0.0007
          }
        ]
      },
      {
        "name": "Together.AI",
        "models": [
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0006,
            "output_price": 0.0006
          },
          {
            "name": "Llama 3 70b",
            "input_price": 0.0009,
            "output_price": 0.0009
          }
        ]
      },
      {
        "name": "Perplexity",
        "models": [
          {
            "name": "Llama 3 70b",
            "input_price": 0.001,
            "output_price": 0.001
          },
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0006,
            "output_price": 0.0006
          }
        ]
      },
      {
        "name": "Cohere",
        "models": [
          {
            "name": "Command R+",
            "input_price": 0.003,
            "output_price": 0.015
          },
          {
            "name": "Command R",
            "input_price": 0.0005,
            "output_price": 0.0015
          }
        ]
      },
      {
        "name": "Deepseek",
        "models": [
          {
            "name": "deepseek-chat",
            "input_price": 0.00014,
            "output_price": 0.00028
          },
          {
            "name": "deepseek-coder",
            "input_price": 0.00014,
            "output_price": 0.00028
          }
        ]
      },
      {
        "name": "Anyscale",
        "models": [
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0005,
            "output_price": 0.0005
          },
          {
            "name": "Llama 3 70b",
            "input_price": 0.001,
            "output_price": 0.001
          }
        ]
      },
      {
        "name": "IBM WatsonX",
        "models": [
          {
            "name": "Llama 3 70b",
            "input_price": 0.0018,
            "output_price": 0.0018
          }
        ]
      },
      {
        "name": "Fireworks",
        "models": [
          {
            "name": "Llama 3 70b",
            "input_price": 0.0009,
            "output_price": 0.0009
          },
          {
            "name": "Mixtral 8x7B",
            "input_price": 0.0005,
            "output_price": 0.0005
          }
        ]
      },
      {
        "name": "01.ai",
        "models": [
          {
            "name": "Yi-Large",
            "input_price": 0.003,
            "output_price": 0.003
          }
        ]
      },
      {
        "name": "Writer",
        "models": [
          {
            "name": "Palmyra X 003",
            "input_price": 0.0075,
            "output_price": 0.0225
          },
          {
            "name": "Palmyra X 32k",
            "input_price": 0.001,
            "output_price": 0.002
          },
          {
            "name": "Palmyra X 002",
            "input_price": 0.001,
            "output_price": 0.002
          },
          {
            "name": "Palmyra X 002 32k",
            "input_price": 0.001,
            "output_price": 0.002
          }
        ]
      }
    ]
  }
```

