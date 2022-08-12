package main

import (
	"fmt"
	_ "image/jpeg"
	"log"
	"math/rand"

	"github.com/ivansuteja96/go-onnxruntime"
)

// LD_LIBRARY_PATH=/usr/local/lib go run predict_example.go
func main() {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_VERBOSE, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	detModel, err := onnxruntime.NewORTSession(ortEnvDet, "../../model.onnx", ortDetSO)
	if err != nil {
		log.Println(err)
		return
	}

	shape1 := []int64{3, 4}
	input1 := randFloats(0, 1, int(shape1[0]*shape1[1]))

	shape2 := []int64{4, 3}
	input2 := randFloats(0, 1, int(shape2[0]*shape2[1]))

	res, err := detModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
		},
		{
			Value: input2,
			Shape: shape2,
		},
	})
	if err != nil {
		log.Println(err)
		return
	}

	if len(res) == 0 {
		log.Println("Failed get result")
		return
	}
	fmt.Printf("Success do predict, shape : %+v, result : %+v\n", res[0].Shape, res[0].Value)
}

func randFloats(min, max float32, n int) []float32 {
	res := make([]float32, n)
	for i := range res {
		res[i] = min + rand.Float32()*(max-min)
	}
	return res
}
