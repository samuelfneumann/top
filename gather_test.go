package top

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestGatherUint8(t *testing.T) {
	inBacking := [][]uint8{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
			18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
	}
	inShapes := [][]int{
		{1, 4, 2},
		{8, 4},
	}

	indicesBacking := [][]int{
		{0, 2, 0, 1},
		{2, 3, 1, 1, 2, 1, 3, 3, 1, 1},
	}
	indicesShapes := [][]int{
		{1, 2, 2},
		{5, 2},
	}

	axis := []int{1, 0}

	outBacking := [][]int{
		{0, 5, 0, 3},
		{8, 13, 4, 5, 8, 5, 12, 13, 4, 5},
		{2, 3, 5, 5, 10, 9, 15, 15, 17, 17},
	}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Uint8,
			inShapes[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Int,
			indicesShapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		indices := tensor.NewDense(
			tensor.Int,
			indicesShapes[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		pred, err := Gather(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}
		if !pred.(*tensor.Dense).Eq(out) {
			t.Errorf("expected:\n%v \nreceived:\n%v", out, pred)
		} else if pred.Dtype() != tensor.Int {
			t.Errorf("expected output to have type %v, but got %v",
				tensor.Int, pred.Dtype())
		}
	}
}

func TestGatherInt(t *testing.T) {
	inBacking := [][]int{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
			18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
	}
	inShapes := [][]int{
		{1, 4, 2},
		{8, 4},
	}

	indicesBacking := [][]int{
		{0, 2, 0, 1},
		{2, 3, 1, 1, 2, 1, 3, 3, 1, 1},
	}
	indicesShapes := [][]int{
		{1, 2, 2},
		{5, 2},
	}

	axis := []int{1, 0}

	outBacking := [][]int{
		{0, 5, 0, 3},
		{8, 13, 4, 5, 8, 5, 12, 13, 4, 5},
		{2, 3, 5, 5, 10, 9, 15, 15, 17, 17},
	}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Int,
			inShapes[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Int,
			indicesShapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		indices := tensor.NewDense(
			tensor.Int,
			indicesShapes[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		pred, err := Gather(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}
		if !pred.(*tensor.Dense).Eq(out) {
			t.Errorf("expected:\n%v \nreceived:\n%v", out, pred)
		}
	}
}

func TestGatherF32(t *testing.T) {
	inBacking := [][]float32{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
			18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
	}
	inShapes := [][]int{
		{1, 4, 2},
		{8, 4},
	}

	indicesBacking := [][]int{
		{0, 2, 0, 1},
		{2, 3, 1, 1, 2, 1, 3, 3, 1, 1},
	}
	indicesShapes := [][]int{
		{1, 2, 2},
		{5, 2},
	}

	axis := []int{1, 0}

	outBacking := [][]float32{
		{0, 5, 0, 3},
		{8, 13, 4, 5, 8, 5, 12, 13, 4, 5},
		{2, 3, 5, 5, 10, 9, 15, 15, 17, 17},
	}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float32,
			inShapes[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Float32,
			indicesShapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		indices := tensor.NewDense(
			tensor.Float32,
			indicesShapes[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		pred, err := Gather(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}
		if !pred.(*tensor.Dense).Eq(out) {
			t.Errorf("expected:\n%v \nreceived:\n%v", out, pred)
		}
	}
}

func TestGatherF64(t *testing.T) {
	inBacking := [][]float64{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
			18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
	}
	inShapes := [][]int{
		{1, 4, 2},
		{8, 4},
	}

	indicesBacking := [][]int{
		{0, 2, 0, 1},
		{2, 3, 1, 1, 2, 1, 3, 3, 1, 1},
	}
	indicesShapes := [][]int{
		{1, 2, 2},
		{5, 2},
	}

	axis := []int{1, 0}

	outBacking := [][]float64{
		{0, 5, 0, 3},
		{8, 13, 4, 5, 8, 5, 12, 13, 4, 5},
		{2, 3, 5, 5, 10, 9, 15, 15, 17, 17},
	}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float64,
			inShapes[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Float64,
			indicesShapes[i],
			tensor.WithBacking(outBacking[i]),
		)

		indices := tensor.NewDense(
			tensor.Float64,
			indicesShapes[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		pred, err := Gather(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}
		if !pred.(*tensor.Dense).Eq(out) {
			t.Errorf("expected:\n%v \nreceived:\n%v", out, pred)
		}
	}
}
