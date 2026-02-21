import os
from tree_sitter_languages import get_parser

class CodeParser:
    def __init__(self):
        self.py_parser = get_parser("python")
        self.cpp_parser = get_parser("cpp")

    def parse_file(self, file_path: str, content: bytes):
        """Returns a list of functions and their direct dependencies."""
        if file_path.endswith(".py"):
            lang = "python"
            parser = self.py_parser
        else:
            # Treat .cpp, .h, .cu, .cuh as C++ for parsing structure
            lang = "cuda" if file_path.endswith((".cu", ".cuh")) else "cpp"
            parser = self.cpp_parser
        
        tree = parser.parse(content)
        functions = []
        
        # Use an iterative stack to prevent RecursionError on massive files
        # Stack stores tuples of (node, current_parent_name)
        stack = [(tree.root_node, None)]
        
        while stack:
            node, parent_name = stack.pop()
            next_parent = parent_name
            
            if node.type == "function_definition":
                func_name = self._get_node_name(node, content)
                func_body = content[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
                
                # Find Calls INSIDE this function
                calls = set()
                self._find_calls(node, content, calls)
                
                functions.append({
                    "name": func_name,
                    "code": func_body,
                    "language": lang,
                    "calls": list(calls),  # List of function names this function calls
                    "parent": parent_name  # Track nesting
                })
                next_parent = func_name

            # Append children in reverse order to maintain left-to-right tree traversal
            for child in reversed(node.children):
                stack.append((child, next_parent))

        return functions

    def _get_node_name(self, node, content):
        """Safely extracts just the function name, ignoring parameters and return types."""
        # 1. Python approach
        name_node = node.child_by_field_name('name')
        if name_node:
            return content[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='ignore')
        
        # 2. C++/CUDA approach
        declarator = node.child_by_field_name('declarator')
        if declarator:
            for child in declarator.children:
                if child.type in ['identifier', 'qualified_identifier', 'field_identifier']:
                    full_name = content[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
                    return full_name.split("::")[-1]
            
            # Fallback if tree-sitter structure varies
            text = content[declarator.start_byte:declarator.end_byte].decode('utf-8', errors='ignore')
            return text.split('(')[0].strip()

        return "unknown"

    def _find_calls(self, root_node, content, found_calls):
        """Iteratively finds function calls and cleans namespaces/attributes."""
        # Iterative stack for finding calls to prevent stack overflows
        stack = [root_node]
        ignore_list = {"print", "len", "range", "malloc", "free", "printf", "sizeof", "main"}
        
        while stack:
            node = stack.pop()
            
            if node.type in ["call", "call_expression"]:
                func_node = node.child_by_field_name("function")
                if func_node:
                    raw_name = content[func_node.start_byte:func_node.end_byte].decode('utf-8', errors='ignore')
                    
                    # CLEANUP: Strip Python object attributes and C++ namespaces
                    clean_name = raw_name.split(".")[-1].split("::")[-1]
                    
                    if clean_name not in ignore_list and len(clean_name) > 1:
                        found_calls.add(clean_name)
            
            # Extend stack with children
            stack.extend(node.children)