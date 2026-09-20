# Chromavertex light pole detection experiments

Image processing experiments for detecting
[Chromavertex](https://github.com/asekerci/chromavertex) light poles.

See AGENTS.md for most info. We're all agents now :)

# Quick start
- install [uv](https://docs.astral.sh/uv/)
- install [just](https://github.com/casey/just)
- then:

```sh
just check          # linting, type checks etc. Everything should pass
just measure        # Run all detectors on the example images.
                    # Measurements are written to measurements/.
                    # Intermediate procesing steps are written to proc_img/.
just cam DETECTOR   # run DETECTOR in real time, using a webcam
```

# Misc
`scratchpads/` is intended for human use. I've been using them to understand how
the agent-written detectors work. A python REPL is very handy when using these
files.

See `prompt-templates/` for some examples of how I've been getting agents to
write detectors.
