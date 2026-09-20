You are working in steps towards a goal: Make a detector that gets
less than 5% max error on all images.

Some detectors are already implemented in ./detectors. They all implement
the Protocol in ./detectors/detector.py

Each detector implementation file starts with a summary of how it works.

A summary of all detectors' performance is at ./measurements/summary.txt

Per-detector performance is in ./measurements. Files are named after their
detector.

There are some ideas for detectors in ./docs/detector-ideas.md

Your task: determine the most promising avenue for reaching the goal.

Rules you MUST follow:
- do not read any image files, or perform any kind of analysis of the images.
  Only use text information available in existing files.

Preferences you SHOULD follow:
- prefer techniques that generalize, rather than focusing on image-specific
  details

If you don't have enough information to determine the next step, propose an
experiment to extract more information from the images, or to test any kind of
image processing ideas that may be useful as part of a detector.

If you're getting stuck thinking, or going around in circles, stop early and
just summarise your thoughts so far.

Write your plan to ./prompts/plan-next.md
