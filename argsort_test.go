package top

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func ExampleArgsort() {
	backing := []float64{1, 5, 0, 3, 9, 8, 4, 6, 7}
	in := tensor.NewDense(
		tensor.Float64,
		[]int{3, 3},
		tensor.WithBacking(backing),
	)

	sortedInd, err := Argsort(in, 1)
	if err != nil {
		panic(err)
	}

	fmt.Println("\nInput data:\n", in)
	fmt.Println("Indices to sort input along axis columns:\n", sortedInd)

	// Output:
	// Input data:
	//   ⎡1  5  0⎤
	//   ⎢3  9  8⎥
	//   ⎣4  6  7⎦

	// Indices to sort input along axis columns:
	//   ⎡2  0  1⎤
	//   ⎢0  2  1⎥
	//   ⎣0  1  2⎦
}

func TestArgsort3D(t *testing.T) {
	backing := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	in := tensor.NewDense(
		tensor.Float64,
		[]int{2, 2, 2},
		tensor.WithBacking(backing),
	)
	out := tensor.NewDense(
		tensor.Int,
		[]int{2, 2, 2},
		tensor.WithBacking([]int{0, 1, 0, 1, 0, 1, 0, 1}),
	)

	sortedInd, err := Argsort(in, 2)
	if err != nil {
		t.Error(err)
	}

	if !sortedInd.Eq(out) {
		t.Errorf("expected \n%v \n\nreceived \n%v", out, sortedInd)
	}
}

func TestArgsort(t *testing.T) {
	// float64 Argsort
	inBacking := [][]float64{
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7},
	}
	outBacking := [][]int{
		{2, 0, 1, 0, 2, 1, 0, 1, 2},
		{0, 0, 0, 1, 2, 2, 2, 1, 1},
		{0, 0, 0, 0, 1, 1, 1, 1},
	}
	shapes := [][]int{
		{3, 3},
		{3, 3},
		{2, 4},
	}
	axis := []int{1, 0, 0}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float64,
			shapes[i],
			tensor.WithBacking(inBacking[i]),
		)
		out := tensor.NewDense(
			tensor.Int,
			shapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		sortedInd, err := Argsort(in, axis[i])
		if err != nil {
			t.Error(err)
		}

		if !sortedInd.Eq(out) {
			t.Errorf("expected \n%v \n\nreceived \n%v", out, sortedInd)
		}
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

	f32SortedInd, err := Argsort(f32In, 1)
	if err != nil {
		t.Error(err)
	}

	if !f32SortedInd.Eq(f32Out) {
		t.Errorf("expected \n%v \n\nreceived \n%v", f32Out, f32SortedInd)
	}

	// int Argsort
	intBacking := [][]int{
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
		{1, 5, 0, 3, 9, 8, 4, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7},
	}
	shapes = [][]int{
		{3, 3},
		{3, 3},
		{2, 4},
	}
	axis = []int{1, 0, 0}
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
		tensor.NewDense(
			tensor.Int,
			shapes[2],
			tensor.WithBacking([]int{0, 0, 0, 0, 1, 1, 1, 1}),
		),
	}

	for i := range intBacking {
		intIn := tensor.NewDense(
			tensor.Int,
			shapes[i],
			tensor.WithBacking(intBacking[i]),
		)
		intSortedInd, err := Argsort(intIn, axis[i])
		if err != nil {
			t.Error(err)
		}

		if !intSortedInd.Eq(intOut[i]) {
			t.Errorf("expected \n%v \n\nreceived \n%v", intOut, intSortedInd)
		}
	}

}
