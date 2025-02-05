package sample

import (
	"cmp"
	"errors"
	"math"
	"slices"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Transform interface {
	Apply([]float64) ([]float64, error)
}

type Sampler interface {
	Sample([]float64) (int, error)
}

type SamplerChain struct {
	transforms []Transform
	sampler    Sampler
}

func NewSampler(transforms []Transform, sampler Sampler) *SamplerChain {
	return &SamplerChain{
		transforms: transforms,
		sampler:    sampler,
	}
}

// TODO(parthsareen): potentially cache softmax values
func softmax(logits []float64) []float64 {
	copiedLogits := make([]float64, len(logits))
	copy(copiedLogits, logits)
	for i := range copiedLogits {
		copiedLogits[i] = math.Exp(copiedLogits[i])
	}

	floatSum := floats.Sum(copiedLogits)
	floats.Scale(1.0/floatSum, copiedLogits)

	return copiedLogits
}

type Temperature float64

func (t Temperature) Apply(logits []float64) ([]float64, error) {
	if t < 0 || t > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}

	// subtracting max logit to avoid under/overflow
	maxLogit := floats.Max(logits)

	temp := math.Max(float64(t), 1e-7)
	for i := range logits {
		logits[i] = (logits[i] - maxLogit) / temp
	}

	return logits, nil
}

type TopK int

// TODO(parthsareen): avoid having to check all logits after this transform
func (k TopK) Apply(logits []float64) ([]float64, error) {
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if int(k) >= len(logits) {
		return logits, nil
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(logits[j], logits[i])
	})

	for _, idx := range indices[k:] {
		logits[idx] = math.Inf(-1)
	}

	return logits, nil
}

type TopP float64

func (p TopP) Apply(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs := softmax(logits)

	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(probs[j], probs[i])
	})

	var cumSum float64
	for i, idx := range indices {
		cumSum += probs[idx]
		if cumSum > float64(p) {
			for _, idx := range indices[i+1:] {
				logits[idx] = math.Inf(-1)
			}
			break
		}
	}
	return logits, nil
}

type MinP float64

func (p MinP) Apply(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs := softmax(logits)
	copiedProbs := make([]float64, len(probs))
	copy(copiedProbs, probs)

	slices.Sort(copiedProbs)

	maxProb := copiedProbs[len(copiedProbs)-1]
	probThreshold := float64(p) * maxProb

	for i := range probs {
		if probs[i] < probThreshold {
			logits[i] = math.Inf(-1)
		}
	}

	return logits, nil
}

type weighed struct {
	src rand.Source
}

func Weighed(seed ...int64) Sampler {
	var src rand.Source
	if len(seed) > 0 {
		src = rand.NewSource(uint64(seed[0]))
	}
	return weighed{src: src}
}

func (s weighed) Sample(logits []float64) (int, error) {
	logitsCopy := make([]float64, 0, len(logits))
	indices := make([]int, 0, len(logits))
	// the uv sampler does not support NaN values
	for i, logit := range logits {
		if !math.IsInf(logit, -1) {
			logitsCopy = append(logitsCopy, logit)
			indices = append(indices, i)
		}
	}

	if len(logitsCopy) == 0 {
		return -1, errors.New("no valid logits found for weighed sampling")
	}

	probs := softmax(logitsCopy)
	w := sampleuv.NewWeighted(probs, s.src)
	if idx, ok := w.Take(); ok {
		return indices[idx], nil
	}
	return -1, errors.New("weighed sampler failed, no valid token found")
}

func (s *SamplerChain) Sample(input []float32) (int, error) {
	logits := make([]float64, len(input))
	for i, v := range input {
		logits[i] = float64(v)
	}

	var err error
	for _, t := range s.transforms {
		if t == Temperature(0) {
			s.sampler = Greedy()
		} else {
			logits, err = t.Apply(logits)
			if err != nil {
				return -1, err
			}
		}
	}

	return s.sampler.Sample(logits)
}
