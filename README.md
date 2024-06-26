# MLX Make More

WIP. Make More with MLX!

makemore takes one text file as input, and predicts more for you. With a wide range of model selections, such as:

- [x] Bigram
- [x] Bigram Neural Network
- [x] Multi-layer Perceptron (MLP)
- [ ] Convolution Nerual Network (CNN)
- [ ] Recurrent Neural Network (RNN)
- [ ] Long Short-Term Memory (LSTM)
- [ ] Gated Recurrent Unit (GRU)
- [ ] Transform

## Plots
### Bigram Frequencies
![image](https://github.com/theSalted/mlx-makemore/assets/30554090/20d04eff-8c8c-4f1b-a3c3-ef5df100307d)

### MLP Word Embedddings (C) Principal Components (1&2)
![MLP Word Feature Embedding](https://github.com/theSalted/mlx-makemore/assets/30554090/792e4d27-7320-4a31-8fe6-ca9249798e5f)

### MLP Learning Rates
![MLP Learning Rates](https://github.com/theSalted/mlx-makemore/assets/30554090/3b710ce7-d8be-4a8b-8204-473dff88739c)

### MLP Losses (log10)
![MLP Log10 Loss Values](https://github.com/theSalted/mlx-makemore/assets/30554090/1cde3335-deae-4916-bbb3-1847f48de4c9)

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
