// Package benchmark provides tools for performance testing of Ollama inference server and supported models.
package benchmark

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"testing"
	"text/tabwriter"
	"time"

	"github.com/ollama/ollama/api"
)

// ServerURL is the default Ollama server URL for benchmarking
const serverURL = "http://127.0.0.1:11434"

// Command line flags
var model string

func init() {
	flag.StringVar(&model, "m", "", "Name of the model to benchmark (required)")
	flag.Lookup("m").DefValue = "model"
}

// metrics collects all benchmark results for final reporting
var metrics []BenchmarkMetrics

type TestCase struct {
	name      string // Human-readable test name
	prompt    string // Input prompt text
	maxTokens int    // Maximum tokens to generate
}

// BenchmarkMetrics contains performance measurements for a single test run
type BenchmarkMetrics struct {
	model           string        // Model being tested
	testName        string        // Name of the test case
	ttft            time.Duration // Time To First Token (TTFT)
	totalTime       time.Duration // Total time for complete response
	totalTokens     int           // Total generated tokens
	tokensPerSecond float64       // Calculated throughput
}

func TestMain(m *testing.M) {
	flag.Parse()
	if model == "" {
		fmt.Fprintln(os.Stderr, "Error: -model flag is required")
		os.Exit(1)
	}
	os.Exit(m.Run())
}

// BenchmarkColdStart runs benchmarks with model loading from cold state
func BenchmarkColdStart(b *testing.B) {
	client := setupBenchmark(b)
	tests := []TestCase{
		{"short_prompt", "Write a long story", 100},
		{"medium_prompt", "Write a detailed economic analysis", 500},
		{"long_prompt", "Write a comprehensive AI research paper", 1000},
	}

	for _, tt := range tests {
		testName := fmt.Sprintf("%s/cold/%s", model, tt.name)
		b.Run(testName, func(b *testing.B) {
			results := make([]BenchmarkMetrics, b.N)

			b.ResetTimer()
			for i := range b.N {
				// Ensure model is unloaded before each iteration
				unloadModel(client, model, b)

				results[i] = runSingleIteration(context.Background(), client, tt, model, b)
			}
			metrics = append(metrics, results...)
		})
	}

	b.Cleanup(func() { reportMetrics(metrics) })
}

// BenchmarkWarmStart runs benchmarks with pre-loaded model
func BenchmarkWarmStart(b *testing.B) {
	client := setupBenchmark(b)
	tests := []TestCase{
		{"short_prompt", "Write a long story", 100},
		{"medium_prompt", "Write a detailed economic analysis", 500},
		{"long_prompt", "Write a comprehensive AI research paper", 1000},
	}

	for _, tt := range tests {
		testName := fmt.Sprintf("%s/warm/%s", model, tt.name)
		b.Run(testName, func(b *testing.B) {
			results := make([]BenchmarkMetrics, b.N)

			// Pre-warm the model
			warmupModel(client, model, tt.prompt, b)

			b.ResetTimer()
			for i := range b.N {
				results[i] = runSingleIteration(context.Background(), client, tt, model, b)
			}
			metrics = append(metrics, results...)
		})
	}

	b.Cleanup(func() { reportMetrics(metrics) })
}

// setupBenchmark verifies server and model availability
func setupBenchmark(b *testing.B) *api.Client {
	resp, err := http.Get(serverURL + "/api/version")
	if err != nil {
		b.Fatalf("Server unavailable: %v", err)
	}
	defer resp.Body.Close()
	b.Log("Server available")

	client := api.NewClient(mustParse(serverURL), http.DefaultClient)
	if _, err := client.Show(context.Background(), &api.ShowRequest{Model: model}); err != nil {
		b.Fatalf("Model unavailable: %v", err)
	}

	return client
}

// warmupModel ensures the model is loaded and warmed up
func warmupModel(client *api.Client, model string, prompt string, b *testing.B) {
	for range 2 {
		err := client.Generate(
			context.Background(),
			&api.GenerateRequest{
				Model:   model,
				Prompt:  prompt,
				Options: map[string]interface{}{"num_predict": 50, "temperature": 0.1},
			},
			func(api.GenerateResponse) error { return nil },
		)
		if err != nil {
			b.Logf("Error during model warm-up: %v", err)
		}
	}
}

// unloadModel forces model unloading using KeepAlive: 0 parameter.
// Includes short delay to ensure unloading completes before next test.
func unloadModel(client *api.Client, model string, b *testing.B) {
	req := &api.GenerateRequest{
		Model:     model,
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := client.Generate(context.Background(), req, func(api.GenerateResponse) error { return nil }); err != nil {
		b.Logf("Unload error: %v", err)
	}
	time.Sleep(100 * time.Millisecond)
}

// runSingleIteration executes a single benchmark iteration
func runSingleIteration(ctx context.Context, client *api.Client, tt TestCase, model string, b *testing.B) BenchmarkMetrics {
	start := time.Now()
	var ttft time.Duration
	var tokens int
	lastToken := start

	req := &api.GenerateRequest{
		Model:   model,
		Prompt:  tt.prompt,
		Options: map[string]interface{}{"num_predict": tt.maxTokens, "temperature": 0.1},
	}

	if b != nil {
		b.Logf("Prompt length: %d chars", len(tt.prompt))
	}

	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if ttft == 0 {
			ttft = time.Since(start)
		}
		if resp.Response != "" {
			tokens++
			lastToken = time.Now()
		}
		return nil
	})
	if err != nil {
		b.Logf("Generation error: %v", err)
	}

	totalTime := lastToken.Sub(start)
	return BenchmarkMetrics{
		model:           model,
		testName:        tt.name,
		ttft:            ttft,
		totalTime:       totalTime,
		totalTokens:     tokens,
		tokensPerSecond: float64(tokens) / totalTime.Seconds(),
	}
}

// reportMetrics processes collected metrics and prints results
func reportMetrics(results []BenchmarkMetrics) {
	if len(results) == 0 {
		return
	}

	type statsKey struct {
		model    string
		testName string
	}
	stats := make(map[statsKey]*struct {
		ttftSum      time.Duration
		totalTimeSum time.Duration
		tokensSum    int
		iterations   int
	})

	for _, m := range results {
		key := statsKey{m.model, m.testName}
		if _, exists := stats[key]; !exists {
			stats[key] = &struct {
				ttftSum      time.Duration
				totalTimeSum time.Duration
				tokensSum    int
				iterations   int
			}{}
		}

		stats[key].ttftSum += m.ttft
		stats[key].totalTimeSum += m.totalTime
		stats[key].tokensSum += m.totalTokens
		stats[key].iterations++
	}

	var averaged []BenchmarkMetrics
	for key, data := range stats {
		count := data.iterations
		averaged = append(averaged, BenchmarkMetrics{
			model:           key.model,
			testName:        key.testName,
			ttft:            data.ttftSum / time.Duration(count),
			totalTime:       data.totalTimeSum / time.Duration(count),
			totalTokens:     data.tokensSum / count,
			tokensPerSecond: float64(data.tokensSum) / data.totalTimeSum.Seconds(),
		})
	}

	printTableResults(averaged)
	printCSVResults(averaged)
}

func printTableResults(averaged []BenchmarkMetrics) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "\nAVERAGED BENCHMARK RESULTS")
	fmt.Fprintln(w, "Model\tTest Name\tTTFT (ms)\tTotal Time (ms)\tTokens\tTokens/sec")
	for _, m := range averaged {
		fmt.Fprintf(w, "%s\t%s\t%.2f\t%.2f\t%d\t%.2f\n",
			m.model,
			m.testName,
			float64(m.ttft.Milliseconds()),
			float64(m.totalTime.Milliseconds()),
			m.totalTokens,
			m.tokensPerSecond,
		)
	}
	w.Flush()
}

func printCSVResults(averaged []BenchmarkMetrics) {
	fmt.Println("\nCSV OUTPUT")
	fmt.Println("model,test_name,ttft_ms,total_ms,tokens,tokens_per_sec")
	for _, m := range averaged {
		fmt.Printf("%s,%s,%.2f,%.2f,%d,%.2f\n",
			m.model,
			m.testName,
			float64(m.ttft.Milliseconds()),
			float64(m.totalTime.Milliseconds()),
			m.totalTokens,
			m.tokensPerSecond,
		)
	}
}

func mustParse(rawURL string) *url.URL {
	u, err := url.Parse(rawURL)
	if err != nil {
		panic(err)
	}
	return u
}
