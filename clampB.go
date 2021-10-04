package top

import (
	"fmt"

	"gorgonia.org/tensor"
)

// ClampB is the backward pass of Clamp. This function works for tensors
// storing float64, float32, or any integer data type. If an integer
// data type is used, the result will be an int tensor regardless of
// the input integer type.
func ClampB(in tensor.Tensor, min, max interface{}) (tensor.Tensor, error) {
	switch in.Dtype() {
	case tensor.Float64:
		_, okMax := max.(float64)
		if !okMax {
			return nil, fmt.Errorf("clampb: data type of max (%T) must "+
				"match data type of in (%v)", max, in.Dtype())
		}
		_, okMin := min.(float64)
		if !okMin {
			return nil, fmt.Errorf("clampb: data type of min (%T) must "+
				"match data type of in (%v)", min, in.Dtype())
		}
		return f64ClampB(in, min.(float64), max.(float64))

	case tensor.Float32:
		_, okMax := max.(float32)
		if !okMax {
			return nil, fmt.Errorf("clampb: data type of max (%T) must "+
				"match data type of in (%v)", max, in.Dtype())
		}
		_, okMin := min.(float32)
		if !okMin {
			return nil, fmt.Errorf("clampb: data type of min (%T) must "+
				"match data type of in (%v)", min, in.Dtype())
		}
		return f32ClampB(in, min.(float32), max.(float32))

	case tensor.Int:
		iMax, err := anyIntToInt(max)
		if err != nil {
			return nil, fmt.Errorf("clampb: could not convert max type %T "+
				"to int", max)
		}
		iMin, err := anyIntToInt(min)
		if err != nil {
			return nil, fmt.Errorf("clampb: could not convert min type %T "+
				"to int", max)
		}
		return intClampB(in, iMin, iMax)

	default:
		return nil, fmt.Errorf("clampb: cannot calculate clampb on tensor "+
			"of type %v", in.Dtype())
	}

}

func f64ClampB(in tensor.Tensor, min, max float64) (tensor.Tensor, error) {
	out := in.Clone().(tensor.Tensor)

	for i := 0; i < out.Size(); i++ {
		at, err := tensor.Itol(i, out.Shape(), out.Strides())
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not compute index: %v", err)
		}
		val, err := out.At(at...)
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not get value at "+
				"coordinates %v: %v", at, err)
		}

		if val.(float64) < min {
			err = out.SetAt(0.0, at...)
		} else if val.(float64) > max {
			err = out.SetAt(0.0, at...)
		} else {
			err = out.SetAt(1.0, at...)
		}
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not clamp at "+
				"coordinates %v: %v", at, err)
		}
	}
	return out, nil
}

func f32ClampB(in tensor.Tensor, min, max float32) (tensor.Tensor, error) {
	out := in.Clone().(tensor.Tensor)

	for i := 0; i < out.Size(); i++ {
		at, err := tensor.Itol(i, out.Shape(), out.Strides())
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not compute index: %v", err)
		}
		val, err := out.At(at...)
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not get value at "+
				"coordinates %v: %v", at, err)
		}

		if val.(float32) < min {
			err = out.SetAt(float32(0.0), at...)
		} else if val.(float32) > max {
			err = out.SetAt(float32(0.0), at...)
		} else {
			err = out.SetAt(float32(1.0), at...)
		}
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not clamp at "+
				"coordinates %v: %v", at, err)
		}
	}
	return out, nil
}

func intClampB(in tensor.Tensor, min, max int) (tensor.Tensor, error) {
	out := tensor.NewDense(tensor.Int, in.Shape())

	for i := 0; i < out.Size(); i++ {
		at, err := tensor.Itol(i, out.Shape(), out.Strides())
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not compute index: %v", err)
		}
		val, err := in.At(at...)
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not get value at "+
				"coordinates %v: %v", at, err)
		}

		intVal, err := anyIntToInt(val)
		if err != nil {
			return nil, fmt.Errorf("intClampB: could not convert %T to int",
				val)
		}

		if intVal < min {
			err = out.SetAt(0, at...)
		} else if intVal > max {
			err = out.SetAt(0, at...)
		} else {
			err = out.SetAt(1, at...)
		}
		if err != nil {
			return nil, fmt.Errorf("f64ClampB: could not clamp at "+
				"coordinates %v: %v", at, err)
		}
	}
	return out, nil
}
