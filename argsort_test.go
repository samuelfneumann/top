package top

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestArgsort(t *testing.T) {
	// float64 Argsort
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

	sortedInd := Argsort(in, 1)

	if !sortedInd.Eq(out) {
		t.Error("invalid sort")
	}

	// float32 Argsort
	f32Backing := []float32{1, 5, 0, 3, 9, 8, 4, 6, 7}
	f32In := tensor.NewDense(
		tensor.Float32,
		[]int{3, 3},
		tensor.WithBacking(f32Backing),
	)
	f32Out := tensor.NewDense(
		tensor.Int,
		[]int{3, 3},
		tensor.WithBacking([]int{2, 0, 1, 0, 2, 1, 0, 1, 2}),
	)

	f32SortedInd := Argsort(f32In, 1)

	if !f32SortedInd.Eq(f32Out) {
		t.Error("invalid sort")
	}

	// int Argsort
	intBacking := [][]int{
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
	}
	shapes := [][]int{
		{3, 3},
		{3, 3},
	}
	axis := []int{1, 0}
	intOut := []*tensor.Dense{
		tensor.NewDense(
			tensor.Int,
			shapes[0],
			tensor.WithBacking([]int{2, 0, 1, 0, 2, 1, 0, 1, 2}),
		),
		tensor.NewDense(
			tensor.Int,
			shapes[1],
			tensor.WithBacking([]int{0, 0, 0, 1, 2, 2, 2, 1, 1}),
		),
	}

	for i := range intBacking {
		intIn := tensor.NewDense(
			tensor.Int,
			shapes[i],
			tensor.WithBacking(intBacking[i]),
		)
		intSortedInd := Argsort(intIn, axis[i])

		if !intSortedInd.Eq(intOut[i]) {
			t.Errorf("expected \n%v \n\nreceived \n%v", intOut, intSortedInd)
		}
	}

}
