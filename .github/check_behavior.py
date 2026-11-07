"""Portable AST fingerprints for the original numerical/manager method bodies."""
import ast
import hashlib
import json


def function_digest(node, module_map):
    prefix='mimic_baseline.tasks.supermimic.'
    class Normalize(ast.NodeTransformer):
        def visit_Import(self,node): return None
        def visit_ImportFrom(self,node): return None
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                for old,new in sorted(module_map.items(),key=lambda x:len(x[1]),reverse=True):
                    node.value=node.value.replace(prefix+new,prefix+old)
            return node
    node=Normalize().visit(ast.parse(ast.unparse(node)))
    def encode(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,[(name,encode(v)) for name,v in ast.iter_fields(value)
                                         if not (name=='type_params' and not v)]]
        if isinstance(value,list): return [encode(v) for v in value]
        if value is Ellipsis: return ["ellipsis"]
        if isinstance(value,bytes): return ["bytes",value.hex()]
        if isinstance(value,complex): return ["complex",repr(value)]
        return value
    return hashlib.sha256(json.dumps(encode(node),sort_keys=True).encode()).hexdigest()
