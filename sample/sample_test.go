package sample

import (
	"fmt"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestTemperature(t *testing.T) {
	logits, err := Temperature(0.5).Apply([]float64{2, -1, 4, -3, 1, -2, 0})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{-4, -10, 0, -14, -6, -12, -8}
	if diff := cmp.Diff(want, logits); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	logits, err = Temperature(-1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for temperature=-1, got %v", logits)
	}
	logits, err = Temperature(2.1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for temperature=2.1, got %v", logits)
	}
}

func TestSoftmax(t *testing.T) {
	probs := softmax([]float64{-3, -2, -1, 0, 1, 2, 4})

	expectedProbs := []float64{0.000751406628089903, 0.0020425349829204676, 0.005552185728064613, 0.015092405572827691, 0.04102541181635154, 0.11151863144543739, 0.8240174238263085}
	if diff := cmp.Diff(expectedProbs, probs); diff != "" {
		t.Errorf("probs mismatch (-want +got):\n%s", diff)
	}
}

func TestTopK(t *testing.T) {
	logits, err := TopK(3).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	expectedlogits := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 1, 2, 4}
	if diff := cmp.Diff(expectedlogits, logits); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = TopK(0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for k=0, got %v", err)
	}

	logits, err = TopK(10).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	expectedlogits = []float64{-3, -2, -1, 0, 1, 2, 4}
	if diff := cmp.Diff(expectedlogits, logits); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}
}

func TestTopP(t *testing.T) {
	logits, err := TopP(0.9).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 2, 4}
	if diff := cmp.Diff(want, logits); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = TopP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Error("expected error for p=1.0")
	}
	_, err = TopP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Error("expected error for p=0.0")
	}
}

func TestMinP(t *testing.T) {
	logits, err := MinP(0.2).Apply([]float64{-3, -2, -1, 0, 1, 2, 4, 3})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 4, 3}
	if diff := cmp.Diff(want, logits); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = MinP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Error("expected error for p=1.0")
	}
	_, err = MinP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Error("expected error for p=0.0")
	}
}

func TestWeighed(t *testing.T) {
	idx, err := Weighted(nil).Sample([]float64{math.Inf(-1), 2, math.Inf(-1), math.Inf(-1)})
	if err != nil {
		t.Error(err)
		return
	}
	want := 1
	if diff := cmp.Diff(want, idx); diff != "" {
		t.Errorf("index mismatch (-want +got):\n%s", diff)
	}

	idx, err = Weighted(nil).Sample([]float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)})
	if err == nil {
		t.Error("expected error for no valid tokens, got index", idx)
	}
}

func TestSample(t *testing.T) {
	input := []float32{1, 2, 3, 4}

	var callOrder []int
	mock1 := &testTransform{
		id:        1,
		callOrder: &callOrder,
	}
	mock2 := &testTransform{
		id:        2,
		callOrder: &callOrder,
	}
	mock3 := &testTransform{
		id:        3,
		callOrder: &callOrder,
	}
	sampler := NewSampler([]Transform{mock1, mock2, mock3}, Greedy())

	got, err := sampler.Sample(input)
	if err != nil {
		t.Error(err)
		return
	}

	wantOrder := []int{1, 2, 3}
	if diff := cmp.Diff(wantOrder, callOrder); diff != "" {
		t.Errorf("call order mismatch (-want +got):\n%s", diff)
	}

	want := 3 // Greedy sampler should pick highest logit
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("sampled index mismatch (-want +got):\n%s", diff)
	}

	errMock := &testTransform{
		returnErr: fmt.Errorf("mock error"),
	}
	sampler = NewSampler([]Transform{mock1, errMock, mock2}, Greedy())
	_, err = sampler.Sample(input)
	if err == nil {
		t.Error("Expected error from sampler")
	}
}

type testTransform struct {
	id        int
	callOrder *[]int
	returnErr error
}

func (ts *testTransform) Apply(logits []float64) ([]float64, error) {
	if ts.callOrder != nil {
		*ts.callOrder = append(*ts.callOrder, ts.id)
	}
	if ts.returnErr != nil {
		return nil, ts.returnErr
	}
	return logits, nil
}

func TestSampleTemperatureZero(t *testing.T) {
	sampler := NewSampler([]Transform{Temperature(0)}, Weighted(nil))
	got, err := sampler.Sample([]float32{1, 2, 3, 4})
	if err != nil {
		t.Error(err)
		return
	}
	want := 3 // Greedy sampler should pick highest logit index
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("sampled index mismatch (-want +got):\n%s", diff)
	}
}
