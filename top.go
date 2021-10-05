package top

import (
	"fmt"
	"os"
	"runtime"

	colour "github.com/samuelfneumann/gocolour"
)

func init() {
	GoArch64Bit := []string{"amd64", "arm64", "arm64be", "loong64",
		"mips64", "mips64le", "ppc64", "ppc64le", "riscv64",
		"s390x", "sparc64", "wasm",
	}

	flag := false
	for _, arch := range GoArch64Bit {
		if arch == runtime.GOARCH {
			flag = true
			break
		}
	}

	if !flag {
		fmt.Fprintf(os.Stderr,
			colour.Red+"WARNING: using 32-bit precision, use caution when "+
				"using top with int64 values which may be cast to int (int32)")
	}
}
