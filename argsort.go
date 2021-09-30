package top

import (
	"sort"

	"gorgonia.org/tensor"
)

type intSlice []int

func (f intSlice) Len() int { return len(f) }

func (f intSlice) Less(i, j int) bool { return f[i] < f[j] }

func (f intSlice) Swap(i, j int) { f[i], f[j] = f[j], f[i] }

type f32Slice []float32

func (f f32Slice) Len() int { return len(f) }

func (f f32Slice) Less(i, j int) bool { return f[i] < f[j] }

func (f f32Slice) Swap(i, j int) { f[i], f[j] = f[j], f[i] }

type f64Slice []float64

func (f f64Slice) Len() int { return len(f) }

func (f f64Slice) Less(i, j int) bool { return f[i] < f[j] }

func (f f64Slice) Swap(i, j int) { f[i], f[j] = f[j], f[i] }

type argSorter struct {
	s   sort.Interface
	ind []int
}

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

func (a *argSorter) Less(i, j int) bool {
	return a.s.Less(a.ind[i], a.ind[j])
}

func (a *argSorter) Len() int { return a.s.Len() }

func (a *argSorter) Swap(i, j int) {
	a.ind[i], a.ind[j] = a.ind[j], a.ind[i]
}

func argSort(s sort.Interface) []int {
	a := newArgSorter(s)
	sort.Stable(a)

	return a.ind
}

func Argsort(t tensor.Tensor, axis int) tensor.Tensor {

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
	data := t.Data().([]float64)

	// Sort each row concurrently passing back a slice of indices that
	// would sort the input tensor along the row, along with the indices
	// at which these values should be set for the backing data of
	// the argsort'd tensor
	sortedArgsChan := make(chan []int)
	indChan := make(chan []int)
	for i := 0; i < dimSize; i++ {
		row := i // Since i may change before the goroutine needs it
		go sortRow(data, indChan, sortedArgsChan, row, outer, oStride,
			dimSize, dimStride)
	}

	// Set each row based on the concurrent argsorts
	sorted := make([]int, dataLen) // Backing for argsort'd tensor
	for k := 0; k < dimSize; k++ {
		indices := <-indChan     // Indices to set in the backing slice
		args := <-sortedArgsChan // Sorted indices of the input slice

		for i := 0; i < len(indices); i++ {
			sorted[indices[i]] = args[i]
		}
	}

	// Construct and returns the argsort'd tensor
	indices := tensor.NewDense(
		tensor.Int,
		shape,
		tensor.WithBacking(sorted),
	)
	return indices
}

func sortRow(data interface{}, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
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
	}
}

func f64SortRow(data []float64, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]float64, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(f64Slice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}

func f32SortRow(data []float32, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]float32, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(f32Slice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}

func intSortRow(data []int, indChan, sortedArgsChan chan []int, row,
	outer, oStride, dimSize, dimStride int) {
	currentRow := make([]int, 0, dimSize)
	indices := make([]int, 0, dimSize)
	for j := row * outer; j < row*outer+oStride; j += dimStride {
		currentRow = append(currentRow, data[j])
		indices = append(indices, j)
	}

	args := argSort(intSlice(currentRow))

	indChan <- indices
	sortedArgsChan <- args
}
