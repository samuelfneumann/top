# top Extended T(ensor) Op(erations)

`top` is a `Go` module to provide extended tensor operations to the
`gorgonia.org/tensor` package. My research requires some functions that
are not included in the tensor package listed above, and this module
provides those function.

The module is intended to be used with floating point tensors, but
can be used with integer tensors as well. Because there are so many
different integer types, this package **always** casts integer types
to `int` before working with them, and the output of any operation
on an integer tensor is a tensor that holds `int` values, regardless
of whether the input held a different integer type. For example, if
`Gather()` is run on a tensor of type `tensor.Uint8`, this module will
return a tensor of type `tensor.Int`. This should be okay, because in
most cases a `tensor.Int` should be used anyway. Because of this
implementation detail, take care when working on a 32-bit machine.
Tensors of type `int64` will be converted to `int32` (the definition
of `int` on a 32-bit machine is `int32`), which can result in
truncation and incorrect results. Use `int` whenever possible when
working on a 32-bit machine.

Once `Go` incorporates generics, this module will be updated to deal
with all numeric types appropriately.
