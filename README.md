# MLX Make More

WIP. Make More with MLX!

makemore takes one text file as input, and predicts more for you. With a wide range of model selections, such as:

- Bigram
- Multi-layer Perceptron (MLP)
- Convolution Nerual Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transform

![BigramMap](https://github.com/theSalted/mlx-makemore/assets/30554090/2f9a89ba-914e-4c61-b713-e7a4bbb38408)

## Build Run in CLI

To run in command line, go to the package directory.

Then `chmod` the helper script:

`chmod +x mlx-run.sh`

Then:

`./mlx-run.sh --package mlx-makemore`


## Troubleshoot

Try build and run project following CLI instruction.

Alternatively run `xcodebuild build -scheme mlx-makemore -destination 'platform=OS X' -derivedDataPath ./.derivedData`

## Reference
See [original python repo](makemore) by one and only [Andrej](https://github.com/karpathy)
