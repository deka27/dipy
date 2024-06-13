import ast
import os
import subprocess

DECORATOR = "warning_for_keywords"

class FunctionTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Check if the function contains multiline strings
        if self.has_multiline_string(node.body):
            return node

        # Check if the function has keyword arguments
        num_args = len(node.args.args)
        num_defaults = len(node.args.defaults)
        has_keyword_arguments = num_defaults > 0

        # Check if *args or **kwargs are present
        has_varargs = node.args.vararg is not None
        has_varkwargs = node.args.kwarg is not None

        # Only proceed if the function doesn't already have *args or **kwargs
        if has_keyword_arguments and not (has_varargs or has_varkwargs):
            new_args = []
            non_default_args_count = num_args - num_defaults

            # Add positional arguments before the asterisk
            new_args.extend(node.args.args[:non_default_args_count])

            # Add the asterisk
            new_args.append(ast.arg(arg="*", annotation=None))

            # Add remaining keyword arguments
            new_args.extend(node.args.args[non_default_args_count:])
            node.args.args = new_args

            # Add decorator
            decorator = ast.Call(
                func=ast.Name(id=DECORATOR, ctx=ast.Load()),
                args=[],
                keywords=[]
            )
            node.decorator_list.insert(0, decorator)
        
        return node

    def has_multiline_string(self, body):
        for child in body:
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.BinOp):
                if isinstance(child.value.left, ast.Str) and isinstance(child.value.right, ast.Str):
                    return True
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Str):
                if '\n' in child.value.s:
                    return True
        return False

def process_file(filename):
    with open(filename, "r") as source:
        source_code = source.read()
    
    tree = ast.parse(source_code)
    transformer = FunctionTransformer()
    transformer.visit(tree)
    tree = ast.fix_missing_locations(tree)

    code = ast.unparse(tree)

    with open(filename, "w") as source:
        source.write(code)
    
    # Reformat the file using black to ensure proper formatting
    subprocess.run(["black", filename, "--skip-string-normalization"])

# Specify the files to process
files_to_process = ["imaffine.py"]  # Add all the files you want to process

for file in files_to_process:
    process_file(file)

print("Processing complete.")