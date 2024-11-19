# run.py

import os

def run_example_1():
    os.system('code2prompt --path /path/to/your/script.py')

def run_example_2():
    os.system('code2prompt --path /path/to/your/project --output project_summary.md')

def run_example_3():
    os.system('code2prompt --path /path/to/src --path /path/to/lib --exclude "*/tests/*" --output codebase_summary.md')

def run_example_4():
    os.system('code2prompt --path /path/to/library --output library_docs.md --suppress-comments --line-number --filter "*.py"')

def run_example_5():
    os.system('code2prompt --path /path/to/project --filter "*.js,*.ts" --exclude "node_modules/*,dist/*" --template code_review.j2 --output code_review.md')

def run_example_6():
    os.system('code2prompt --path /path/to/src/components --suppress-comments --tokens --encoding cl100k_base --output ai_input.md')

def run_example_7():
    os.system('code2prompt --path /path/to/project --template comment_density.j2 --output comment_analysis.md --filter "*.py,*.js,*.java"')

def run_example_8():
    os.system('code2prompt --path /path/to/important_file1.py --path /path/to/important_file2.js --line-number --output critical_files.md')

if __name__ == "__main__":
    # Run all examples
    run_example_1()
    run_example_2()
    run_example_3()
    run_example_4()
    run_example_5()
    run_example_6()
    run_example_7()
    run_example_8()