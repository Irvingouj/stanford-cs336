# CS336 Practice Guide

Use the side-by-side folders:

```text
course/assignments/<unit>/
├── answer/     your editable copy
└── tutorial/   installed course material
```

## Assignments

| Unit | Work here | Read here |
| --- | --- | --- |
| 1 Basics | `course/assignments/01-basics/answer` | `course/assignments/01-basics/tutorial` |
| 2 Systems | `course/assignments/02-systems/answer` | `course/assignments/02-systems/tutorial` |
| 3 Scaling | `course/assignments/03-scaling/answer` | `course/assignments/03-scaling/tutorial` |
| 4 Data | `course/assignments/04-data/answer` | `course/assignments/04-data/tutorial` |
| 5 Alignment | `course/assignments/05-alignment/answer` | `course/assignments/05-alignment/tutorial` |

## Lectures

```text
course/lectures
course/official
```

## Run Checks

Each assignment keeps its own `pyproject.toml`, tests, scripts, and lockfile.

```bash
cd course/assignments/01-basics/answer
uv run pytest
```
