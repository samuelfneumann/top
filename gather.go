package top

import (
	"fmt"

	"gorgonia.org/tensor"
)

// Gather gathers values along axis at the indices specified by indices.
// The indices tensor must have the same number of dimensions as t
// and must have any integer (e.g. int, uint, in16, ...) backing data.
//
// For a 3D tensor, the output is specified by:
//
//		out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
//		out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
//		out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
//
//
// Gather works on tensors t of type float64, float32, or any int type.
// If the backing data of a tensor is an int type (e.g. uint32), it
// will be converted to an int in the returned tensor.
//
// The indices tensor must store an integer type.
// Regardless of the integer type stored in the indices tensor, this
// function will convert that integer type to an int before computing
// the gather function. If using a 32-bit machine, use caution
// if the data type stored by the indices tensor is int64, as this
// may result in trucation or numerical issues.
//
// This implementation is heavily based on the PyTorch implementation.
// See PyTorch's documentation for more details and usage:
// https://pytorch.org/docs/stable/generated/torch.gather.html
func Gather(t tensor.Tensor, axis int, indices tensor.Tensor) (tensor.Tensor,
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
			"the same number of dimensions but got indices=(%v) and t=(%v)",
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
		return gatherF64(t, axis, indices)

	case tensor.Float32:
		return gatherF32(t, axis, indices)

	case tensor.Int, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
		tensor.Uint, tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64:
		return gatherInt(t, axis, indices)

	default:
		return nil, fmt.Errorf("gather: cannot gather on tensor of type %v",
			t.Dtype())
	}
}

// gatherF64 gathers elements from a float64 tensor. See Gather for
// more details.
func gatherF64(t tensor.Tensor, axis int,
	indices tensor.Tensor) (tensor.Tensor, error) {
	// Backing data
	output := make([]float64, indices.Size())

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

		out, err := t.At(coords...)
		if err != nil {
			return nil, fmt.Errorf("gather: could not get element at index %v",
				ijk)
		}
		output[i] = out.(float64)
	}

	return tensor.NewDense(
		tensor.Float64,
		indices.Shape(),
		tensor.WithBacking(output),
	), nil
}

// gatherF32 gathers elements from a float32 tensor. See Gather for
// more details.
func gatherF32(t tensor.Tensor, axis int,
	indices tensor.Tensor) (tensor.Tensor, error) {
	// Backing data
	output := make([]float32, indices.Size())

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

		out, err := t.At(coords...)
		if err != nil {
			return nil, fmt.Errorf("gather: could not get element at index %v",
				ijk)
		}
		output[i] = out.(float32)
	}

	return tensor.NewDense(
		tensor.Float32,
		indices.Shape(),
		tensor.WithBacking(output),
	), nil
}

// gatherInt gathers elements from any int tensor. The tensor may have
// any integer type backing data (e.g. uint, uint32, int64, ...), but
// the resulting tensor will always have backing type int (other integer
// types will be cast to int) See Gather for more details.
func gatherInt(t tensor.Tensor, axis int,
	indices tensor.Tensor) (tensor.Tensor, error) {
	// Backing data
	output := make([]int, indices.Size())

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

		out, err := t.At(coords...)
		if err != nil {
			return nil, fmt.Errorf("gather: could not get element at index %v",
				ijk)
		}

		elem, err := anyIntToInt(out)
		if err != nil {
			return nil, fmt.Errorf("gatherInt: could not convert type %T to "+
				"int", out)
		}
		output[i] = elem
	}

	return tensor.NewDense(
		tensor.Int,
		indices.Shape(),
		tensor.WithBacking(output),
	), nil
}
