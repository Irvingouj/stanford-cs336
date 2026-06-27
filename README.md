# Stanford CS336 Meta Repo

Everything for the class lives here, split by purpose:

```text
course/       side-by-side map: answer/ and tutorial/ for each assignment
answers/      your editable working copies
tutorials/    installed course material: assignments, lectures, official PDFs
reference/    vendored source archive used by tutorials/
.ref/         upstream clones, kept hidden
```

Start in `course/assignments`:

```text
course/assignments/01-basics/
├── answer/     -> your editable assignment 1 copy
└── tutorial/   -> installed course assignment 1 reference
```

The same shape exists for:

```text
01-basics
02-systems
03-scaling
04-data
05-alignment
```

Lectures and handouts:

```text
course/lectures   -> tutorials/lectures
course/official   -> tutorials/official
```

## Work Loop

```bash
cd course/assignments/01-basics/answer
uv run pytest
```

Read the matching tutorial next door:

```bash
open ../tutorial/cs336_assignment1_basics.pdf
```
