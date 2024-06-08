# MLX Package Template

For simple [MLX](https://github.com/ml-explore/mlx-swift) playground projects, `.xcodeproj` is overkill and `.playground` don't support dependencies. A Swift Package is your best bet, but configure it can be a hassle. 

So use this template as a fast and easy way to play with [MLX](https://github.com/ml-explore/mlx-swift). 

- Note: MLX Require Xcode to build Metal. And if you are just get started, please use Xcode.

## Rename the Template

Just rename everything named `mlx-package-template`. Most importantly, the project directory name, and all mentions in `Package.swift`

You can freely name `Template.swift` and `struct Template` to anything you like. 


## Build Run in CLI

To run in command line, go to the package directory.

Then `chmod` the helper script:

`chmod +x mlx-run.sh`

Then:

`./mlx-run.sh --package mlx-package-template`

(If you renamed project, remember to change `mlx-package-template` arg).


## Troubleshoot

Try build and run project following CLI instruction.

Alternatively run `xcodebuild build -scheme mlx-package-template -destination 'platform=OS X' -derivedDataPath ./.derivedData`

(If you renamed project, remember to change `mlx-package-template` arg).
