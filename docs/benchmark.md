# Benchmark

Go benchmark tests that measure end-to-end performance of a running Ollama server. Run these tests to evaluate model inference performance on your hardware and measure the impact of code changes.

## When to use

Run these benchmarks when:
- Making changes to the model inference engine
- Modifying model loading/unloading logic
- Changing prompt processing or token generation code
- Implementing a new model architecture
- Testing performance across different hardware setups

## Prerequisites
- Ollama server running locally (`127.0.0.1:11434`)
- benchstat tool: `go install golang.org/x/perf/cmd/benchstat@latest`

## Usage and Examples

**Note**: All commands must be run from the `benchmark` directory of the Ollama project.

Basic syntax:
```bash
go test -bench=. -m MODEL_NAME ./...
```

Required flags:
- `-m`: Model name to benchmark
- `-bench=.`: Run all benchmarks

Optional flags:
- `-count N`: Number of times to run the benchmark (useful for statistical analysis)
- `-timeout T`: Maximum time for the benchmark to run (e.g. "10m" for 10 minutes)

Common usage patterns:

Single benchmark run:
```bash
go test -bench=. -m mixtral:7b ./...
```

Compare two different models:
```bash
# First run
go test -bench=. -count 10 -m llama2:7b ./... > bench1.txt

# Second run
go test -bench=. -count 10 -m mistral:7b ./... > bench2.txt

# Compare results using benchstat
benchstat bench1.txt bench2.txt
```

## Output metrics

The benchmark reports several key metrics:

- `gen_tok/s`: Generated tokens per second
- `prompt_tok/s`: Prompt processing tokens per second
- `ttft_ms`: Time to first token in milliseconds
- `load_ms`: Model load time in milliseconds
- `gen_tokens`: Total tokens generated
- `prompt_tokens`: Total prompt tokens processed

Each benchmark runs two scenarios:
- Cold start: Model is loaded from disk for each test
- Warm start: Model is pre-loaded in memory

Three prompt lengths are tested for each scenario:
- Short prompt (100 tokens)
- Medium prompt (500 tokens)
- Long prompt (1000 tokens)
