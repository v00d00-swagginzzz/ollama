package sample

import "gonum.org/v1/gonum/floats"

type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

func (s greedy) Sample(logits []float32, transforms ...Transform) (int, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	// Tranforms are not applied here, as greedy sampling is just max logit index
	return floats.MaxIdx(logits64), nil
}
