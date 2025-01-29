# Benchmark

Performance benchmarking for Ollama.

## Prerequisites
- Ollama server running locally (`127.0.0.1:11434`)

## Run Benchmark
```bash
# Run all tests
go test -bench=. -m $MODEL_NAME -timeout 30m ./...
```
