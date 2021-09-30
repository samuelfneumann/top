package top

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestArgsort(t *testing.T) {
	backing := []float64{1, 5, 0, 3, 9, 8, 4, 6, 7}
	in := tensor.NewDense(
		tensor.Float64,
		[]int{3, 3},
		tensor.WithBacking(backing),
	)
	out := tensor.NewDense(
		tensor.Int,
		[]int{3, 3},
		tensor.WithBacking([]int{2, 0, 1, 0, 2, 1, 0, 1, 2}),
	)

	sorted_ind := Argsort(in, 1)

	if !sorted_ind.Eq(out) {
		t.Error("invalid sort")
	}
}
