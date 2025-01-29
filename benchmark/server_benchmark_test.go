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
var modelFlag string

func init() {
	flag.StringVar(&modelFlag, "m", "", "Name of the model to benchmark (required)")
	flag.Lookup("m").DefValue = "model" // Set the default value for usage display
}

// metrics collects all benchmark results for final reporting
var metrics []BenchmarkMetrics

// TestCase defines a benchmark test scenario with prompt characteristics
type TestCase struct {
	name      string // Human-readable test name
	prompt    string // Input prompt text
	maxTokens int    // Maximum tokens to generate
}

// BenchmarkMetrics contains performance measurements for a single test run
type BenchmarkMetrics struct {
	model           string        // Model being tested
	scenario        string        // cold_start or warm_start
	testName        string        // Name of the test case
	ttft            time.Duration // Time To First Token (TTFT)
	totalTime       time.Duration // Total time for complete response
	totalTokens     int           // Total generated tokens
	tokensPerSecond float64       // Calculated throughput
}

// ScenarioType defines the initialization state for benchmarking
type ScenarioType int

const (
	ColdStart ScenarioType = iota // Model is loaded from cold state
	WarmStart                     // Model is already loaded in memory
)

// String implements fmt.Stringer for ScenarioType
func (s ScenarioType) String() string {
	return [...]string{"cold_start", "warm_start"}[s]
}

func TestMain(m *testing.M) {
	flag.Parse()
	if modelFlag == "" {
		fmt.Fprintln(os.Stderr, "Error: -model flag is required")
		os.Exit(1)
	}
	os.Exit(m.Run())
}

// BenchmarkServerInference is the main entry point for benchmarking Ollama inference performance.
// It uses the model specified by the -model flag.
func BenchmarkServerInference(b *testing.B) {
	b.Logf("Starting benchmark suite for model: %s", modelFlag)

	// Verify server availability
	resp, err := http.Get(serverURL + "/api/version")
	if err != nil {
		b.Fatalf("Server unavailable: %v", err)
	}
	defer resp.Body.Close()
	b.Log("Server available")

	tests := []TestCase{
		{"short_prompt", "Write a long story", 100},
		{"medium_prompt", "Write a detailed economic analysis", 500},
		{"long_prompt", "Write a comprehensive AI research paper", 1000},
	}

	// Register cleanup handler for results reporting
	b.Cleanup(func() { reportMetrics(metrics) })

	client := api.NewClient(mustParse(serverURL), http.DefaultClient)
	// Verify model availability
	if _, err := client.Show(context.Background(), &api.ShowRequest{Model: modelFlag}); err != nil {
		b.Fatalf("Model unavailable: %v", err)
	}

	for _, tt := range tests {
		testName := fmt.Sprintf("%s/%s/%s", modelFlag, ColdStart, tt.name)
		b.Run(testName, func(b *testing.B) {
			m := runBenchmark(b, tt, modelFlag, ColdStart, client)
			metrics = append(metrics, m...)
		})
	}

	for _, tt := range tests {
		testName := fmt.Sprintf("%s/%s/%s", modelFlag, WarmStart, tt.name)
		b.Run(testName, func(b *testing.B) {
			m := runBenchmark(b, tt, modelFlag, WarmStart, client)
			metrics = append(metrics, m...)
		})
	}
}

// runBenchmark executes multiple iterations of a specific test case and scenario.
// Returns collected metrics for all iterations.
func runBenchmark(b *testing.B, tt TestCase, model string, scenario ScenarioType, client *api.Client) []BenchmarkMetrics {
	results := make([]BenchmarkMetrics, b.N)

	if scenario == WarmStart {
		// Pre-warm the model by generating some tokens
		for range 2 {
			err := client.Generate(
				context.Background(),
				&api.GenerateRequest{
					Model:   model,
					Prompt:  tt.prompt,
					Options: map[string]interface{}{"num_predict": 50, "temperature": 0.1},
				},
				func(api.GenerateResponse) error { return nil },
			)
			if err != nil {
				b.Logf("Error during model warm-up: %v", err)
			}
		}
	}

	b.ResetTimer()

	// Run benchmark iterations
	for i := range b.N {
		if scenario == ColdStart {
			unloadModel(client, model, b)
		}

		results[i] = runSingleIteration(context.Background(), client, tt, model, b)
		results[i].scenario = scenario.String()
	}
	return results
}

// unloadModel forces model unloading using KeepAlive: -1 parameter.
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

// runSingleIteration measures performance metrics for a single inference request.
// Captures TTFT, total generation time, and calculates tokens/second.
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

	// Execute generation request with metrics collection
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

// reportMetrics processes collected metrics and prints formatted results.
// Generates both human-readable tables and CSV output with averaged statistics.
func reportMetrics(results []BenchmarkMetrics) {
	if len(results) == 0 {
		return
	}

	// Aggregate results by test case
	type statsKey struct {
		model    string
		scenario string
		testName string
	}
	stats := make(map[statsKey]*struct {
		ttftSum      time.Duration
		totalTimeSum time.Duration
		tokensSum    int
		iterations   int
	})

	for _, m := range results {
		key := statsKey{m.model, m.scenario, m.testName}
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

	// Calculate averages
	var averaged []BenchmarkMetrics
	for key, data := range stats {
		count := data.iterations
		averaged = append(averaged, BenchmarkMetrics{
			model:           key.model,
			scenario:        key.scenario,
			testName:        key.testName,
			ttft:            data.ttftSum / time.Duration(count),
			totalTime:       data.totalTimeSum / time.Duration(count),
			totalTokens:     data.tokensSum / count,
			tokensPerSecond: float64(data.tokensSum) / data.totalTimeSum.Seconds(),
		})
	}

	// Print formatted results
	printTableResults(averaged)
	printCSVResults(averaged)
}

// printTableResults displays averaged metrics in a formatted table
func printTableResults(averaged []BenchmarkMetrics) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "\nAVERAGED BENCHMARK RESULTS")
	fmt.Fprintln(w, "Model\tScenario\tTest Name\tTTFT (ms)\tTotal Time (ms)\tTokens\tTokens/sec")
	for _, m := range averaged {
		fmt.Fprintf(w, "%s\t%s\t%s\t%.2f\t%.2f\t%d\t%.2f\n",
			m.model,
			m.scenario,
			m.testName,
			float64(m.ttft.Milliseconds()),
			float64(m.totalTime.Milliseconds()),
			m.totalTokens,
			m.tokensPerSecond,
		)
	}
	w.Flush()
}

// printCSVResults outputs averaged metrics in CSV format
func printCSVResults(averaged []BenchmarkMetrics) {
	fmt.Println("\nCSV OUTPUT")
	fmt.Println("model,scenario,test_name,ttft_ms,total_ms,tokens,tokens_per_sec")
	for _, m := range averaged {
		fmt.Printf("%s,%s,%s,%.2f,%.2f,%d,%.2f\n",
			m.model,
			m.scenario,
			m.testName,
			float64(m.ttft.Milliseconds()),
			float64(m.totalTime.Milliseconds()),
			m.totalTokens,
			m.tokensPerSecond,
		)
	}
}

// mustParse is a helper function to parse URLs with panic on error
func mustParse(rawURL string) *url.URL {
	u, err := url.Parse(rawURL)
	if err != nil {
		panic(err)
	}
	return u
}
