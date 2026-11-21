#!/usr/bin/env python3
"""零依赖 JSON Schema 子集校验器（技能库快照数据自校验用）。

支持关键字子集：type（含多类型数组/["x","null"]）、required、properties、
items、enum、const、minLength、minimum、maximum、minItems、pattern、
additionalProperties(bool)。不实现的键一律忽略（宽松降级，不误报）。

用法: python json_schema_mini.py <schema.json> <data.json> [data2.json ...]
退出码: 0=全部通过  1=存在校验错误  2=用法/文件错误
"""
import json
import re
import sys

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is_type(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, TYPES[t])


def validate(value, schema, path, errors):
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(value, x) for x in types):
            errors.append("%s: type 应为 %s，实际 %s" % (path, types, type(value).__name__))
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append("%s: 值 %r 不在 enum %s" % (path, value, schema["enum"]))
    if "const" in schema and value != schema["const"]:
        errors.append("%s: 值 %r != const %r" % (path, value, schema["const"]))
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errors.append("%s: 缺必填键 %r" % (path, k))
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in value:
                validate(value[k], sub, "%s.%s" % (path, k), errors)
        ap = schema.get("additionalProperties")
        if ap is False:
            extra = set(value) - set(props)
            if extra:
                errors.append("%s: 出现未声明键 %s" % (path, sorted(extra)))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("%s: 数组长度 %d < minItems %d" % (path, len(value), schema["minItems"]))
        items = schema.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(value):
                validate(el, items, "%s[%d]" % (path, i), errors)
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("%s: 字符串长度 %d < minLength %d" % (path, len(value), schema["minLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append("%s: %r 不匹配 pattern %s" % (path, value, schema["pattern"]))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("%s: %r < minimum %r" % (path, value, schema["minimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("%s: %r > maximum %r" % (path, value, schema["maximum"]))


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(2)
    try:
        schema = json.load(open(sys.argv[1], encoding="utf-8"))
    except Exception as e:
        print("schema 读取失败: %r" % e, file=sys.stderr)
        sys.exit(2)
    bad = 0
    for dp in sys.argv[2:]:
        try:
            data = json.load(open(dp, encoding="utf-8"))
        except Exception as e:
            print("FAIL %s: JSON 解析失败 %r" % (dp, e))
            bad += 1
            continue
        errors = []
        validate(data, schema, "$", errors)
        if errors:
            bad += 1
            print("FAIL %s: %d 处不符 schema" % (dp, len(errors)))
            for msg in errors[:20]:
                print("  " + msg)
            if len(errors) > 20:
                print("  ...（共 %d 条）" % len(errors))
        else:
            print("PASS %s" % dp)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
