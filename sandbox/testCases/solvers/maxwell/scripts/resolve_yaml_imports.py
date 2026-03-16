#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return {} if data is None else data


def to_import_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("import must be a string or list of strings")


def resolve_node(node, current_file, stack):
    if isinstance(node, dict):
        imported_merged = {}
        if "import" in node:
            import_paths = to_import_list(node["import"])
            for rel in import_paths:
                import_file = (current_file.parent / rel).resolve()
                if import_file in stack:
                    cycle = " -> ".join(str(p) for p in list(stack) + [import_file])
                    raise ValueError(f"Import cycle detected: {cycle}")
                imported_node = load_yaml(import_file)
                resolved_import = resolve_node(imported_node, import_file, stack | {import_file})
                if not isinstance(resolved_import, dict):
                    raise ValueError(f"Imported file must resolve to a mapping: {import_file}")
                imported_merged = deep_merge(imported_merged, resolved_import)

        local = {}
        for key, value in node.items():
            if key == "import":
                continue
            local[key] = resolve_node(value, current_file, stack)

        return deep_merge(imported_merged, local)

    if isinstance(node, list):
        return [resolve_node(item, current_file, stack) for item in node]

    return node


def main():
    parser = argparse.ArgumentParser(
        description="Resolve custom YAML 'import' keys into a plain YAML file."
    )
    parser.add_argument("input", help="Path to source YAML file with import keys")
    parser.add_argument("-o", "--output", required=True, help="Path to resolved YAML output")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2

    root = load_yaml(input_path)
    resolved = resolve_node(root, input_path, {input_path})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("%YAML 1.1\n---\n")
        yaml.safe_dump(resolved, stream, sort_keys=False, default_flow_style=False)
        stream.write("...\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
