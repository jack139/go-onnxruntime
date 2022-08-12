package main

import (
	"fmt"
	"os"
	_ "image/jpeg"
	"log"
	"math/rand"

	"github.com/ivansuteja96/go-onnxruntime"
	"github.com/disintegration/imaging"
)

// LD_LIBRARY_PATH=/usr/local/lib go run predict_example2.go
func main() {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_VERBOSE, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	detModel, err := onnxruntime.NewORTSession(ortEnvDet, "../../../multinfer/data/det_10g.onnx", ortDetSO)
	if err != nil {
		log.Println(err)
		return
	}

	shape1 := []int64{1, 3, 640, 640}
	input1 := preprocessImage("../../5.jpg", 640)
	//input1 := randFloats(0, 1, int(shape1[0]*shape1[1]*shape1[2]*shape1[3]))


	res, err := detModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
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

	for i:=0;i<len(res);i++ {
		fmt.Printf("Success do predict, shape : %+v, result : %+v\n", 
			res[i].Shape, 
			res[i].Value.([]float32)[0], // only show one value
		)
	}
}


func randFloats(min, max float32, n int) []float32 {
	res := make([]float32, n)
	for i := range res {
		res[i] = min + rand.Float32()*(max-min)
	}
	return res
}



func Transpose(rgbs []float32) []float32 {
	out := make([]float32, len(rgbs))
	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		out[i] = rgbs[i*3]
		out[i+channelLength] = rgbs[i*3+1]
		out[i+channelLength*2] = rgbs[i*3+2]
	}
	return out
}

func preprocessImage(imageFile string, inputSize int) []float32 {
	src, err := imaging.Open(imageFile)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
		os.Exit(1)
	}

	rgbs := make([]float32, inputSize*inputSize*3)

	result := imaging.Resize(src, inputSize, inputSize, imaging.Lanczos)
	//result = imaging.CropAnchor(result, 224, 224, imaging.Center)
	j := 0
	for i := range result.Pix {
		if (i+1)%4 != 0 {
			rgbs[j] = float32(result.Pix[i])
			j++
		}
	}

	rgbs = Transpose(rgbs)
	//channelLength := len(rgbs) / 3
	//for i := 0; i < channelLength; i++ {
	//	rgbs[i] = normalize(rgbs[i]/255, 0.485, 0.229)
	//	rgbs[i+channelLength] = normalize(rgbs[i+channelLength]/255, 0.456, 0.224)
	//	rgbs[i+channelLength*2] = normalize(rgbs[i+channelLength*2]/255, 0.406, 0.225)
	//}
	return rgbs
}

