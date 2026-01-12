import json
import re

input_path = "movies.json"
output_path = "movies_fixed.json"

pattern = re.compile(
    r"^(?P<body>.+),\s*(?P<article>The|A|An)(?P<rest>\s*\(.*\))?\s*$",
    flags=re.IGNORECASE,
)


def fix_title(title):
    if not isinstance(title, str):
        return title

    match = pattern.match(title.strip())
    if not match:
        return title

    body = match.group("body").strip()
    article = match.group("article").capitalize()
    rest = match.group("rest") or ""

    return f"{article} {body}{rest}"


with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_count = 0

for row in data:
    if isinstance(row, list) and len(row) > 1:
        original = row[1]
        fixed = fix_title(original)
        if fixed != original:
            row[1] = fixed
            fixed_count += 1

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. Fixed {fixed_count} titles.")
