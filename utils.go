package top

import (
	"fmt"
	"math/rand"
)

// randInt returns a random int slice of length size
func randInt(size int, min, max int) []int {
	slice := make([]int, size)
	for i := range slice {
		slice[i] = min + rand.Intn(max-min)
	}

	return slice
}

// anyIntToInt converts any integer type to int
func anyIntToInt(integer interface{}) (int, error) {
	switch i := integer.(type) {
	case int:
		return i, nil
	case uint:
		return int(i), nil
	case uint8:
		return int(i), nil
	case uint16:
		return int(i), nil
	case uint32:
		return int(i), nil
	case uint64:
		return int(i), nil
	case int8:
		return int(i), nil
	case int16:
		return int(i), nil
	case int32:
		return int(i), nil
	case int64:
		return int(i), nil
	default:
		return 0, fmt.Errorf("anyIntToInt: input type %T is not an integer "+
			"type", integer)
	}
}
