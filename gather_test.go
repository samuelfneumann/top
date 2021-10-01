package top

import (
	"testing"

	"gorgonia.org/tensor"
)

// TestGatherArgsort tests if argsort'ng a tensor, then gathering along
// the argsort'd indices results in a sorted tensor
func TestGatherArgsort(t *testing.T) {
	inBacking := [][]float64{
		{10, 89, 31, 12, 243, 53, 1, 32},
		{1239, 123, 15, 123, 903, 64, 31, 62, 51, 12, 31, 1154},
		{1},
		{1},
		{
			0.6814, 0.0457, 0.9451, 0.3334, 0.2042, 0.6429, 0.1355, 0.2120,
			0.8991, 0.7385, 0.3468, 0.8606, 0.6819, 0.1200, 0.2257, 0.0314,
			0.2968, 0.0756, 0.0122, 0.2243, 0.2294, 0.0123, 0.1043, 0.1117,
			0.6476, 0.3482, 0.2128, 0.4513, 0.8428, 0.6201, 0.4108, 0.4702,
			0.9517, 0.4650, 0.3424, 0.5208, 0.6751, 0.4491, 0.9173, 0.6082,
			0.3168, 0.8348, 0.1936, 0.2932, 0.8929, 0.9531, 0.8600, 0.3467,
			0.6973, 0.3509, 0.0340, 0.0792, 0.3808, 0.8463, 0.9289, 0.0178,
			0.8601, 0.5616, 0.8342, 0.8999, 0.3162, 0.7145, 0.6970, 0.0997,
		},
		{
			0.4916, 0.8388, 0.0596, 0.4493, 0.7331, 0.0806, 0.0479, 0.7520,
			0.8535, 0.5718, 0.5433, 0.5492, 0.3948, 0.9193, 0.9723, 0.2726,
			0.8944, 0.2635, 0.2940, 0.7699, 0.5824, 0.5608, 0.6324, 0.8291,
			0.7676, 0.2119, 0.8638, 0.7051, 0.3790, 0.0070, 0.4403, 0.6703,
			0.6500, 0.7430, 0.6955, 0.0861,
		},
		{
			0.9068, 0.7465, 0.5014, 0.5224, 0.0377, 0.6265,
		},
	}

	outBacking := [][]float64{
		{10, 89, 12, 31, 53, 243, 1, 32},
		{15, 123, 1239, 123, 31, 62, 903, 64, 31, 12, 51, 1154},
		{1},
		{1},
		{
			0.0457, 0.3334, 0.6814, 0.9451, 0.1355, 0.2042, 0.2120, 0.6429,
			0.3468, 0.7385, 0.8606, 0.8991, 0.0314, 0.1200, 0.2257, 0.6819,
			0.0122, 0.0756, 0.2243, 0.2968, 0.0123, 0.1043, 0.1117, 0.2294,
			0.2128, 0.3482, 0.4513, 0.6476, 0.4108, 0.4702, 0.6201, 0.8428,
			0.3424, 0.4650, 0.5208, 0.9517, 0.4491, 0.6082, 0.6751, 0.9173,
			0.1936, 0.2932, 0.3168, 0.8348, 0.3467, 0.8600, 0.8929, 0.9531,
			0.0340, 0.0792, 0.3509, 0.6973, 0.0178, 0.3808, 0.8463, 0.9289,
			0.5616, 0.8342, 0.8601, 0.8999, 0.0997, 0.3162, 0.6970, 0.7145,
		},
		{
			0.0596, 0.0806, 0.4493, 0.4916, 0.7331, 0.8388, 0.0479, 0.5433,
			0.5492, 0.5718, 0.7520, 0.8535, 0.2635, 0.2726, 0.3948, 0.8944,
			0.9193, 0.9723, 0.2940, 0.5608, 0.5824, 0.6324, 0.7699, 0.8291,
			0.0070, 0.2119, 0.3790, 0.7051, 0.7676, 0.8638, 0.0861, 0.4403,
			0.6500, 0.6703, 0.6955, 0.7430,
		},
		{
			0.0377, 0.5014, 0.5224, 0.6265, 0.7465, 0.9068,
		},
	}

	axis := []int{1, 1, 1, 0, 2, 1, 0}

	shape := [][]int{
		{4, 2},
		{3, 2, 2},
		{1},
		{1},
		{4, 4, 4},
		{6, 6},
		{6},
	}

	errorExpected := []bool{false, false, true, false, false, false, false}

	for i := range inBacking {
		in := tensor.NewDense(
			tensor.Float64,
			shape[i],
			tensor.WithBacking(inBacking[i]),
		)

		out := tensor.NewDense(
			tensor.Float64,
			shape[i],
			tensor.WithBacking(outBacking[i]),
		)

		argSorted, err := Argsort(in, axis[i])
		if err != nil && !errorExpected[i] {
			t.Error(err)
			continue
		} else if errorExpected[i] {
			continue
		}

		gathered, err := Gather(in, axis[i], argSorted)
		if err != nil && !errorExpected[i] {
			t.Error(err)
			continue
		} else if errorExpected[i] {
			continue
		}

		if !gathered.Eq(out) {
			t.Errorf("expected: \n%v \nreceived: \n%v", out, gathered)
		}
	}
}

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
