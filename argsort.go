package top

import (
	"fmt"
	"sort"

	"gorgonia.org/tensor"
)

// Argsort returns an int tensor containing indices that would sort
// t along axis.
func Argsort(t tensor.Tensor, axis int) (tensor.Tensor, error) {
	// Ensure valid data type of tensor
	switch t.Data().(type) {
	case []float64, []float32, []int:

	default:
		return nil, fmt.Errorf("argsort: unknown tensor type %v", t.Dtype())
	}

	if axis >= len(t.Shape()) {
		return nil, fmt.Errorf("argsort: axis out of range [%v] for "+
			"tensor with %v dimensions", axis, len(t.Shape()))
	}

	shape := make([]int, len(t.Shape()))
	copy(shape, t.Shape())
	reps := tensor.ProdInts(append(shape[:axis], shape[axis+1:]...))

	// sortedInd is the channel along which the argsort'd indices for a
	// specific row will be sent
	sortedInd := make(chan []int, reps)

	// backingInd is the channel along which the indices in the
	// backing slice of the final tensor at which data from
	// sortedInd should be placed is sent. That is, if `b` is sent along
	// backingInd and `s` along sortedInd, then `s[i]` should be placed
	// at index `b[i]` in the backing data of the final, argsort'd
	// tensor
	backingInd := make(chan []int, reps)

	// errors is the channel along which any errors in concurrent
	// argsort'ng are sent
	errors := make(chan error, reps)

	// Sort each row concurrently passing back a slice of indices that
	// would sort the input tensor along the row, along with the indices
	// at which these values should be set for the backing data of
	// the argsort'd tensor
	for i := 0; i < reps; i++ {
		row := i // Since i may change before the goroutine needs it
		go sortRow(t, backingInd, sortedInd, row, axis, errors)
	}

	// Set each row based on the concurrent argsorts
	sorted := make([]int, t.Size()) // Backing for argsort'd tensor
	for k := 0; k < reps; k++ {
		indices := <-backingInd // Indices to set in the backing slice
		args := <-sortedInd     // Sorted indices of the input slice
		err := <-errors         // Errors during sorting
		if err != nil {
			return nil, fmt.Errorf("argsort: %v", err)
		}

		for i := 0; i < len(indices); i++ {
			sorted[indices[i]] = args[i]
		}
	}
	close(backingInd)
	close(sortedInd)
	close(errors)

	// Construct and returns the argsort'd tensor
	indices := tensor.NewDense(
		tensor.Int,
		t.Shape(),
		tensor.WithBacking(sorted),
	)
	return indices, nil
}

// argSorter argsorts a slice
type argSorter struct {
	s   sort.Interface
	ind []int
}

// newArgSorter returns a new argSorter
func newArgSorter(s sort.Interface) *argSorter {
	indices := make([]int, s.Len())
	for i := 0; i < s.Len(); i++ {
		indices[i] = i
	}

	return &argSorter{
		s:   s,
		ind: indices,
	}
}

// Less implements the interface sort.Interface
func (a *argSorter) Less(i, j int) bool {
	return a.s.Less(a.ind[i], a.ind[j])
}

// Len implements the interface sort.Interface
func (a *argSorter) Len() int { return a.s.Len() }

// Swap implements the interface sort.Interface
func (a *argSorter) Swap(i, j int) {
	a.ind[i], a.ind[j] = a.ind[j], a.ind[i]
}

// argSort returns the indices that would sort s
func argSort(s sort.Interface) []int {
	a := newArgSorter(s)
	sort.Stable(a)

	return a.ind
}

// sortRow sorts as specific row of data, sending the indices to update
// in the backing slice of the argsort'd tensor along backingInd, and
// the arguments that sort the tensor along sortedInd. Any errors
// during the computation are sent along errors.
//
// The parameter row indicates which row should be sorted along axis.
// For example, if we want to access the first row along axis 0
// for a tensor of size (2, 2, 2), then these indices will be
// (0, 0, 0), (1, 0, 0). The parameter row actually refers to
// (x, 0, 0) in this case, where x ∈ {0, 1}.
func sortRow(data tensor.Tensor, backingInd, sortedInd chan []int, row,
	axis int, errors chan error) {
	switch data.Data().(type) {
	case []float64:
		f64SortRow(data, backingInd, sortedInd, row, axis, errors)

	case []float32:
		f32SortRow(data, backingInd, sortedInd, row, axis, errors)

	case []int:
		intSortRow(data, backingInd, sortedInd, row, axis, errors)

	default:
		errors <- fmt.Errorf("sortRow: unknown tensor type %v", data.Dtype())
	}
}

func getStaticRowIndices(data tensor.Tensor, row, axis int) ([]int, error) {
	shapes := data.Shape()
	if row > tensor.ProdInts(shapes) {
		return nil, fmt.Errorf("getStaticRowIndices: index out of range [%v] "+
			"for row length %v", row, tensor.ProdInts(shapes))
	}
	static := make([]int, len(data.Shape()))
	current := 0
	for i := 0; i < row; i++ {
		if current == axis {
			// Go to next dimension
			current = (current + 1) % len(shapes)
		}
		static[current]++

		for static[current] == shapes[current] {
			static[current] = 0
			current = (current + 1) % len(shapes)
			if current == axis {
				current = (current + 1) % len(shapes)
			}
			static[current]++
		}
		current = 0
	}
	return static, nil
}

// f64SortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func f64SortRow(data tensor.Tensor, backingInd, sortedInd chan []int, row,
	axis int, errors chan error) {
	// Get the indices for the row along the dimensions different from the
	// sorted axis. These will be static indices for the row, and only
	// the indices along axis will change.
	//
	// For example, if we want to access the first row along axis 0
	// for a tensor of size (2, 2, 2), then these indices will be
	// (0, 0, 0), (1, 0, 0). The parameter row actually refers to
	// (x, 0, 0) in this case, where x ∈ {0, 1}.
	static, err := getStaticRowIndices(data, row, axis)
	if err != nil {
		errors <- fmt.Errorf("f64SortRow: %v", err)

		backingInd <- nil
		sortedInd <- nil
		return
	}

	dimSize := data.Shape()[axis]
	currentRow := make([]float64, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for i := 0; i < dimSize; i++ {
		// Set the index of the next element along the current axis
		static[axis] = i
		// Get the index into the backing slice of the tensor
		j, err := tensor.Ltoi(data.Shape(), data.Strides(), static...)
		if err != nil {
			errors <- fmt.Errorf("f64SortRow: could not compute index "+
				"of coordinates %v into backing slice", static)

			backingInd <- nil
			sortedInd <- nil
			return
		}

		// Construct the current row and store which axis in the
		// backing slice this element is at. This index will be
		// needed to reconstruct the argsort'd row in the
		// final argsort'd tensor.
		currentRow = append(currentRow, data.Data().([]float64)[j])
		indices = append(indices, j)
	}

	// Argsort this row only. These argsort'd indices will be placed at
	// indices (variable above) in the backing slice of the final tensor
	args := argSort(sort.Float64Slice(currentRow))

	// Send the argsort'd indices, along with the indices at which to
	// place them in the backing slice of the final tensor to the
	// main goroutine.
	sortedInd <- args
	backingInd <- indices
	errors <- nil
}

// f32SortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func f32SortRow(data tensor.Tensor, backingInd, sortedInd chan []int, row,
	axis int, errors chan error) {
	// Get the indices for the row along the dimensions different from the
	// sorted axis. These will be static indices for the row, and only
	// the indices along axis will change.
	//
	// For example, if we want to access the first row along axis 0
	// for a tensor of size (2, 2, 2), then these indices will be
	// (0, 0, 0), (1, 0, 0). The parameter row actually refers to
	// (x, 0, 0) in this case, where x ∈ {0, 1}.
	static, err := getStaticRowIndices(data, row, axis)
	if err != nil {
		errors <- fmt.Errorf("f32SortRow: %v", err)

		backingInd <- nil
		sortedInd <- nil
		return
	}

	dimSize := data.Shape()[axis]
	currentRow := make([]float32, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for i := 0; i < dimSize; i++ {
		// Set the index of the next element along the current axis
		static[axis] = i

		// Get the index into the backing slice of the tensor
		j, err := tensor.Ltoi(data.Shape(), data.Strides(), static...)
		if err != nil {
			errors <- fmt.Errorf("f32SortRow: could not compute index "+
				"of coordinates %v into backing slice", static)

			backingInd <- nil
			sortedInd <- nil
			return
		}

		// Construct the current row and store which axis in the
		// backing slice this element is at. This index will be
		// needed to reconstruct the argsort'd row in the
		// final argsort'd tensor.
		currentRow = append(currentRow, data.Data().([]float32)[j])
		indices = append(indices, j)
	}

	// Argsort this row only. These argsort'd indices will be placed at
	// indices (variable above) in the backing slice of the final tensor
	args := argSort(float32Slice(currentRow))

	// Send the argsort'd indices, along with the indices at which to
	// place them in the backing slice of the final tensor to the
	// main goroutine.
	sortedInd <- args
	backingInd <- indices
	errors <- nil
}

// intSortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func intSortRow(data tensor.Tensor, backingInd, sortedInd chan []int, row,
	axis int, errors chan error) {
	// Get the indices for the row along the dimensions different from the
	// sorted axis. These will be static indices for the row, and only
	// the indices along axis will change.
	//
	// For example, if we want to access the first row along axis 0
	// for a tensor of size (2, 2, 2), then these indices will be
	// (0, 0, 0), (1, 0, 0). The parameter row actually refers to
	// (x, 0, 0) in this case, where x ∈ {0, 1}.
	static, err := getStaticRowIndices(data, row, axis)
	if err != nil {
		errors <- fmt.Errorf("intSortRow: %v", err)

		backingInd <- nil
		sortedInd <- nil
		return
	}

	dimSize := data.Shape()[axis]
	currentRow := make([]int, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for i := 0; i < dimSize; i++ {
		// Set the index of the next element along the current axis
		static[axis] = i

		// Get the index into the backing slice of the tensor
		j, err := tensor.Ltoi(data.Shape(), data.Strides(), static...)
		if err != nil {
			errors <- fmt.Errorf("intSortRow: could not compute index "+
				"of coordinates %v into backing slice", static)

			backingInd <- nil
			sortedInd <- nil
			return
		}

		// Construct the current row and store which axis in the
		// backing slice this element is at. This index will be
		// needed to reconstruct the argsort'd row in the
		// final argsort'd tensor.
		currentRow = append(currentRow, data.Data().([]int)[j])
		indices = append(indices, j)
	}

	// Argsort this row only. These argsort'd indices will be placed at
	// indices (variable above) in the backing slice of the final tensor
	args := argSort(sort.IntSlice(currentRow))

	// Send the argsort'd indices, along with the indices at which to
	// place them in the backing slice of the final tensor to the
	// main goroutine.
	sortedInd <- args
	backingInd <- indices
	errors <- nil
}

// float32Slice is a []float32 wrapper to implement sort.Interface
type float32Slice []float32

// Len implements the interface sort.Interface
func (s float32Slice) Len() int { return len(s) }

// Less implements the interface sort.Interface
func (s float32Slice) Less(i, j int) bool { return s[i] < s[j] }

// Swap implements the interface sort.Interface
func (s float32Slice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
