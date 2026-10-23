"""Static transitive local numerical imports, canonical names separate from paths."""
import ast,hashlib
from pathlib import Path
import common
def source_manifest():
    own=Path(__file__).resolve().parent;V=own.parents[1];roots=[('asset_level_v6',V),('methodology_src',common.CODE/'src'),('methodology_scripts',common.CODE/'scripts'),('campaign',common.CAMPAIGN)]
    def resolve(name):
        for _,root in roots:
            base=root.joinpath(*name.split('.'))
            for path in(base.with_suffix('.py'),base/'__init__.py'):
                if path.is_file():return path.resolve()
    pending=[own/'candidate.py',own/'finite_wealth.py',Path(__file__).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path is None or path in seen:continue
        seen.add(path)
        label,root=next((label,root.resolve())for label,root in roots if root.resolve()in path.parents)
        relative=path.relative_to(root);package=list(relative.parts[:-1]);parent=path.parent
        while root==parent or root in parent.parents:
            init=parent/'__init__.py'
            if init.is_file()and init not in seen:pending.append(init)
            if parent==root:break
            parent=parent.parent
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or''
                if node.level:prefix='.'.join(package[:len(package)-node.level+1]+([prefix]if prefix else[]))
                names=([prefix]if prefix else[])+[prefix+'.'+n.name if prefix else n.name for n in node.names]
            pending.extend(resolve(name)for name in names)
    result={}
    for path in sorted(seen):
        label,root=next((label,root.resolve())for label,root in roots if root.resolve()in path.parents)
        result[label+'/'+str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result
