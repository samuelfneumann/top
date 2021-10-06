package top

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestGatherBF64(t *testing.T) {
	inBacking := [][]float64{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
		},
	}

	indicesBacking := [][]int{
		{0, 1, 2, 1},
		{0, 1, 2, 1},
		{0, 1, 2},
		{0, 1, 2},
		{1, 2, 0, 1},
		{0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1},
	}

	targetBacking := [][]float64{
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0},
		{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
		{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0},
		{
			1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
			0, 0, 0, 0,
		},
	}

	inSize := [][]int{
		{4, 3},
		{4, 3},
		{4, 3},
		{4, 3},
		{2, 2, 3},
		{4, 2, 3},
	}

	indicesSize := [][]int{
		{4, 1},
		{2, 2},
		{1, 3},
		{3, 1},
		{2, 2, 1},
		{3, 2, 3},
	}

	axis := []int{1, 1, 0, 0, 2, 2}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float64,
			inSize[i],
			tensor.WithBacking(inBacking[i]),
		)
		target := tensor.NewDense(
			tensor.Float64,
			inSize[i],
			tensor.WithBacking(targetBacking[i]),
		)
		indices := tensor.NewDense(
			tensor.Int,
			indicesSize[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		output, err := GatherB(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}

		if !output.Eq(target) {
			t.Errorf("expected: \n%v \nreceived: \n%v", target, output)
		}
	}
}

func TestGatherBF32(t *testing.T) {
	inBacking := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
		},
	}

	indicesBacking := [][]int{
		{0, 1, 2, 1},
		{0, 1, 2, 1},
		{0, 1, 2},
		{0, 1, 2},
		{1, 2, 0, 1},
		{0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1},
	}

	targetBacking := [][]float32{
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0},
		{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
		{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0},
		{
			1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
			0, 0, 0, 0,
		},
	}

	inSize := [][]int{
		{4, 3},
		{4, 3},
		{4, 3},
		{4, 3},
		{2, 2, 3},
		{4, 2, 3},
	}

	indicesSize := [][]int{
		{4, 1},
		{2, 2},
		{1, 3},
		{3, 1},
		{2, 2, 1},
		{3, 2, 3},
	}

	axis := []int{1, 1, 0, 0, 2, 2}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float32,
			inSize[i],
			tensor.WithBacking(inBacking[i]),
		)
		target := tensor.NewDense(
			tensor.Float32,
			inSize[i],
			tensor.WithBacking(targetBacking[i]),
		)
		indices := tensor.NewDense(
			tensor.Int,
			indicesSize[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		output, err := GatherB(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}

		if !output.Eq(target) {
			t.Errorf("expected: \n%v \nreceived: \n%v", target, output)
		}
	}
}

func TestGatherBInt(t *testing.T) {
	inBacking := [][]int{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
		},
	}

	indicesBacking := [][]int{
		{0, 1, 2, 1},
		{0, 1, 2, 1},
		{0, 1, 2},
		{0, 1, 2},
		{1, 2, 0, 1},
		{0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1},
	}

	targetBacking := [][]int{
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0},
		{1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
		{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0},
		{
			1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
			0, 0, 0, 0,
		},
	}

	inSize := [][]int{
		{4, 3},
		{4, 3},
		{4, 3},
		{4, 3},
		{2, 2, 3},
		{4, 2, 3},
	}

	indicesSize := [][]int{
		{4, 1},
		{2, 2},
		{1, 3},
		{3, 1},
		{2, 2, 1},
		{3, 2, 3},
	}

	axis := []int{1, 1, 0, 0, 2, 2}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Int,
			inSize[i],
			tensor.WithBacking(inBacking[i]),
		)
		target := tensor.NewDense(
			tensor.Int,
			inSize[i],
			tensor.WithBacking(targetBacking[i]),
		)
		indices := tensor.NewDense(
			tensor.Int,
			indicesSize[i],
			tensor.WithBacking(indicesBacking[i]),
		)

		output, err := GatherB(in, axis[i], indices)
		if err != nil {
			t.Error(err)
		}

		if !output.Eq(target) {
			t.Errorf("expected: \n%v \nreceived: \n%v", target, output)
		}
	}
}
