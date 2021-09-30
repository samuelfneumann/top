package top

import (
	"fmt"
	"sort"

	"gorgonia.org/tensor"
)

// Argsort returns an int tensor containing indices that would sort
// t along axis.
func Argsort(t tensor.Tensor, axis int) (tensor.Tensor, error) {
	shape := t.Shape()
	dimSize := shape[axis]

	// Outer is the number of elements to step to get to the next row
	// along the argument axis
	outer := tensor.ProdInts([]int(shape[:axis]))
	if outer == 0 {
		outer = 1
	}

	// The number of elements between consecutive elements along the
	// argument axis
	inner := tensor.ProdInts([]int(shape[axis+1:]))
	if inner == 0 {
		inner = 1
	}
	dimStride := inner

	// oStride is such that slice[row:row+oStride] will contain all
	// the elements of the current row
	oStride := dimSize * dimStride

	dataLen := shape.TotalSize()
	data := t.Data()

	// Sort each row concurrently passing back a slice of indices that
	// would sort the input tensor along the row, along with the indices
	// at which these values should be set for the backing data of
	// the argsort'd tensor
	sortedArgsChan := make(chan []int, outer*inner)
	indChan := make(chan []int, outer*inner)
	errors := make(chan error, outer*inner)
	for i := 0; i < outer*inner; i++ {
		row := i // Since i may change before the goroutine needs it
		go sortRow(data, indChan, sortedArgsChan, row, outer, oStride,
			dimSize, dimStride, errors)
	}

	// Set each row based on the concurrent argsorts
	sorted := make([]int, dataLen) // Backing for argsort'd tensor
	for k := 0; k < inner*outer; k++ {
		indices := <-indChan     // Indices to set in the backing slice
		args := <-sortedArgsChan // Sorted indices of the input slice
		err := <-errors          // Errors during sorting
		if err != nil {
			return nil, fmt.Errorf("argsort: %v", err)
		}

		for i := 0; i < len(indices); i++ {
			sorted[indices[i]] = args[i]
		}
	}
	close(indChan)
	close(sortedArgsChan)
	close(errors)

	// Construct and returns the argsort'd tensor
	indices := tensor.NewDense(
		tensor.Int,
		shape,
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
// in the backing slice of the argsort'd tensor along indChan, and
// the arguments that sort the tensor along sortedArgsChan. Any errors
// during the computation are sent along errors.
func sortRow(data interface{}, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int, errors chan error) {
	switch d := data.(type) {
	case []float64:
		f64SortRow(d, indChan, sortedArgsChan, row, outer, oStride,
			dimSize, dimStride)

	case []float32:
		f32SortRow(d, indChan, sortedArgsChan, row, outer, oStride,
			dimSize, dimStride)

	case []int:
		intSortRow(d, indChan, sortedArgsChan, row, outer, oStride,
			dimSize, dimStride)

	default:
		errors <- fmt.Errorf("sortRow: type %T not supported", data)
	}

	errors <- nil
}

// f64SortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func f64SortRow(data []float64, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]float64, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(sort.Float64Slice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}

// f32SortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func f32SortRow(data []float32, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]float32, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(float32Slice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}

// intSortRow sorts a row of a tensor, where data is the backing slice
// of the tensor. See sortRow.
func intSortRow(data []int, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]int, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(sort.IntSlice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}

// float32Slice is a []float32 wrapper to implement sort.Interface
type float32Slice []float32

// Len implements the interface sort.Interface
func (s float32Slice) Len() int { return len(s) }

// Less implements the interface sort.Interface
func (s float32Slice) Less(i, j int) bool { return s[i] < s[j] }

// Swap implements the interface sort.Interface
func (s float32Slice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
