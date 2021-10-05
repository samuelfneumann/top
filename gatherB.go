package top

import (
	"fmt"

	"gorgonia.org/tensor"
)

// GatherB is the backpropagation of Gather. The input tensor t must
// store either float64's or float32's since it is not possible to
// take a gradient over the integers. The indices tensor must store
// any integer type.
//
// Regardless of the integer type stored in the indices tensor, this
// function will convert that integer type to an int before computing
// the gather backpropagation. If using a 32-bit machine, use caution
// if the data type stored by the indices tensor is int64, as this
// may result in trucation or numerical issues.
func GatherB(t tensor.Tensor, axis int, indices tensor.Tensor) (tensor.Tensor,
	error) {
	// Ensure indices is a tensor of int
	switch indices.Data().(type) {
	case []int, []uint, []uint8, []uint32, []uint64, []uint16,
		[]int8, []int16, []int32, []int64:

	default:
		return nil, fmt.Errorf("gather: unknown indices type %v",
			indices.Dtype())
	}

	// Ensure indices and t have same number of dimensions
	if len(t.Shape()) != len(indices.Shape()) {
		return nil, fmt.Errorf("gather: indices and t tensors must have "+
			"the same number of dimensions but got indices(%v) and t(%v)",
			len(indices.Shape()), len(t.Shape()))
	}

	// Ensure all dimension are legal
	for i := range t.Shape() {
		if i == axis {
			continue
		}
		if t.Shape()[i] < indices.Shape()[i] {
			return nil, fmt.Errorf("gather: size does not match at "+
				"dimension %v expected indices shape %v to be smaller "+
				"than t shape %v apart from dimension %v", i, indices.Shape(),
				t.Shape(), axis)
		}
	}

	// Ensure the axis is legal
	if axis >= len(indices.Shape()) {
		return nil, fmt.Errorf("gather: axis out of range [%v] for "+
			"tensor with %v dimensions", axis, len(indices.Shape()))
	}

	switch t.Dtype() {
	case tensor.Float64:
		return gatherBF64(t, axis, indices)

	case tensor.Float32:
		return gatherBF32(t, axis, indices)

	default:
		return nil, fmt.Errorf("gather: cannot gather on tensor of type %v",
			t.Dtype())
	}
}

func gatherBF64(t tensor.Tensor, axis int,
	indices tensor.Tensor) (tensor.Tensor, error) {
	// Backing data
	output := tensor.NewDense(
		t.Dtype(),
		t.Shape(),
	)

	// Loop through each index in indices
	for i := 0; i < indices.Size(); i++ {
		ijk, err := tensor.Itol(i, indices.Shape(), indices.Strides())
		if err != nil {
			panic(err)
		}

		coords := make([]int, len(ijk))
		copy(coords, ijk)
		index, err := indices.At(ijk...)
		if err != nil {
			panic(err)
		}

		// Convert any int type to int
		intIndex, err := anyIntToInt(index)
		if err != nil {
			return nil, fmt.Errorf("gather: could not get index from indices "+
				"at coordinates %v: %v", coords, err)
		}
		coords[axis] = intIndex

		err = output.SetAt(1.0, coords...)
		if err != nil {
			return nil, fmt.Errorf("gatherB: could not set element at index %v",
				ijk)
		}
	}

	return output, nil
}

func gatherBF32(t tensor.Tensor, axis int,
	indices tensor.Tensor) (tensor.Tensor, error) {
	// Backing data
	output := tensor.NewDense(
		t.Dtype(),
		t.Shape(),
	)

	// Loop through each index in indices
	for i := 0; i < indices.Size(); i++ {
		ijk, err := tensor.Itol(i, indices.Shape(), indices.Strides())
		if err != nil {
			panic(err)
		}

		coords := make([]int, len(ijk))
		copy(coords, ijk)
		index, err := indices.At(ijk...)
		if err != nil {
			panic(err)
		}

		// Convert any int type to int
		intIndex, err := anyIntToInt(index)
		if err != nil {
			return nil, fmt.Errorf("gather: could not get index from indices "+
				"at coordinates %v: %v", coords, err)
		}
		coords[axis] = intIndex

		err = output.SetAt(1.0, coords...)
		if err != nil {
			return nil, fmt.Errorf("gatherB: could not set element at index %v",
				ijk)
		}
	}

	return output, nil
}
