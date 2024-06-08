#!/bin/sh

# Wrapper to help run command line tools -- this will find the build directory
# and set the DYLD_FRAMEWORK_PATH so that command line tools that link frameworks
# can be run.

# Compare to the one from mlx-swift-example, this one is revised to support package


#
# Example:
# ./mlx-run --debug llm-tool --help
#!/bin/sh

# Wrapper to help run command line tools -- this will find the build directory
# and set the DYLD_FRAMEWORK_PATH so that command line tools that link frameworks
# can be run.
#
# Example:
# ./mlx-run --debug llm-tool --help

if [ "$#" -lt 1 ]; then
	echo "usage: mlx-run [--package <package-name>] [--debug/--release] <tool-name> arguments"
	exit 1
fi

CONFIGURATION=Release
PACKAGE_MODE=false

if [ "$1" == "--package" ]; then
	CONFIGURATION=Package
	PACKAGE_NAME="$2"
	PACKAGE_MODE=true
	shift
	shift
fi
if [ "$1" == "--release" ]; then
	CONFIGURATION=Release
	shift
fi
if [ "$1" == "--debug" ]; then
	CONFIGURATION=Debug
	shift
fi
if [ "$1" == "--list" ]; then
	xcodebuild -list
	exit 0
fi

COMMAND="$1"
shift

if [ "$PACKAGE_MODE" == true ]; then
	xcodebuild build -scheme "$PACKAGE_NAME" -destination 'platform=OS X' -derivedDataPath ./.derivedData
	BUILD_DIR=".derivedData/Build/Products/Debug"
	if [ -f "$BUILD_DIR/$PACKAGE_NAME" ]; then
		"$BUILD_DIR/$PACKAGE_NAME" "$@"
		if [ $? -eq 0 ]; then
			echo "\n\033[1m** RUN SUCCEEDED **\033[0m"
		else
			echo "Run failed"
			exit 1
		fi
	else
		echo "$BUILD_DIR/$PACKAGE_NAME does not exist -- check build configuration ($CONFIGURATION)"
		exit 1
	fi
	exit 0
fi

BUILD_DIR=`xcodebuild -configuration $CONFIGURATION -showBuildSettings -scheme $COMMAND | grep 'BUILT_PRODUCTS_DIR = /' | sed -e 's/^[^=]*= //g'`

if [ -d "$BUILD_DIR/$COMMAND.app" ]; then
	exec $BUILD_DIR/$COMMAND.app/Contents/MacOS/$COMMAND "$@" &
fi

if [ -f "$BUILD_DIR/$COMMAND" ]; then
	export DYLD_FRAMEWORK_PATH=$BUILD_DIR/PackageFrameworks:$BUILD_DIR
	exec "$BUILD_DIR/$COMMAND" "$@"
else
	echo "$BUILD_DIR/$COMMAND does not exist -- check build configuration ($CONFIGURATION)"
	exit 1
fi