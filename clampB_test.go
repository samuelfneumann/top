package top

import (
	"math/rand"
	"testing"
	"time"

	"gorgonia.org/tensor"
)

// TestF64ClampB tests the ClampB function on tensors of type float64
func TestF64ClampB(t *testing.T) {
	const numTests int = 15     // The number of random tests to run
	const clipScale float64 = 2 // Legal ranges generated based on clipScale
	const scale float64 = 5     // Values are generated based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		numDims := rand.Intn(dimMax-dimMin) + dimMin
		size := randInt(numDims, sizeMin, sizeMax)

		clampMin := clipScale * (rand.Float64() - 1) // [-clipScale, 0)
		clampMax := clipScale * rand.Float64()       // [0, clipScale)

		min := clampMin * scale
		max := clampMax * scale

		inBacking := make([]float64, tensor.ProdInts(size))
		outBacking := make([]float64, len(inBacking))
		for j := range inBacking {
			inBacking[j] = min + rand.Float64()*(max-min)
			if inBacking[j] < clampMin || inBacking[j] > clampMax {
				outBacking[j] = 0.0
			} else {
				outBacking[j] = 1.0
			}
		}

		in := tensor.NewDense(
			tensor.Float64,
			size,
			tensor.WithBacking(inBacking),
		)

		output, err := ClampB(in, clampMin, clampMax)
		if err != nil {
			t.Error(err)
		}

		data := output.Data().([]float64)
		for i := range data {
			if data[i] != outBacking[i] {
				t.Errorf("expected: %v \nreceived: %v \nindex: %d", data[i],
					outBacking[i], i)
			}
		}
	}
}

// TestF32ClampB tests the ClampB function on tensors of type float32
func TestF32ClampB(t *testing.T) {
	const numTests int = 15     // The number of random tests to run
	const clipScale float32 = 2 // Legal ranges generated based on clipScale
	const scale float32 = 5     // Values are generated based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		numDims := rand.Intn(dimMax-dimMin) + dimMin
		size := randInt(numDims, sizeMin, sizeMax)

		clampMin := clipScale * (rand.Float32() - 1) // [-clipScale, 0)
		clampMax := clipScale * rand.Float32()       // [0, clipScale)

		min := clampMin * scale
		max := clampMax * scale

		inBacking := make([]float32, tensor.ProdInts(size))
		outBacking := make([]float32, len(inBacking))
		for j := range inBacking {
			inBacking[j] = min + rand.Float32()*(max-min)
			if inBacking[j] < clampMin || inBacking[j] > clampMax {
				outBacking[j] = 0.0
			} else {
				outBacking[j] = 1.0
			}
		}

		in := tensor.NewDense(
			tensor.Float32,
			size,
			tensor.WithBacking(inBacking),
		)

		output, err := ClampB(in, clampMin, clampMax)
		if err != nil {
			t.Error(err)
		}

		data := output.Data().([]float32)
		for i := range data {
			if data[i] != outBacking[i] {
				t.Errorf("expected: %v \nreceived: %v \nindex: %d", data[i],
					outBacking[i], i)
			}
		}
	}
}

// TestIntClampB tests the ClampB function on tensors of type int
func TestIntClampB(t *testing.T) {
	const numTests int = 15  // The number of random tests to run
	const clipScale int = 10 // Legal ranges generated based on clipScale
	const scale int = 5      // Values are generated based on scale

	// Randomly generated input has number of dimensions betwee dimMin
	// and dimMax. Each dimension of the randomly generated input has
	// between sizeMin and sizeMax elements.
	const sizeMin int = 1
	const sizeMax int = 10
	const dimMin int = 1
	const dimMax int = 4
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTests; i++ {
		numDims := rand.Intn(dimMax-dimMin) + dimMin
		size := randInt(numDims, sizeMin, sizeMax)

		clampMin := -rand.Intn(clipScale) // [-clipScale, 0)
		clampMax := rand.Intn(clipScale)  // [0, clipScale)

		min := clampMin * scale
		max := clampMax * scale

		inBacking := make([]int, tensor.ProdInts(size))
		outBacking := make([]int, len(inBacking))
		for j := range inBacking {
			inBacking[j] = min + rand.Intn(max-min)
			if inBacking[j] < clampMin || inBacking[j] > clampMax {
				outBacking[j] = 0.0
			} else {
				outBacking[j] = 1.0
			}
		}

		in := tensor.NewDense(
			tensor.Int,
			size,
			tensor.WithBacking(inBacking),
		)

		output, err := ClampB(in, clampMin, clampMax)
		if err != nil {
			t.Error(err)
		}

		data := output.Data().([]int)
		for i := range data {
			if data[i] != outBacking[i] {
				t.Errorf("expected: %v \nreceived: %v \nindex: %d", data[i],
					outBacking[i], i)
			}
		}
	}
}
