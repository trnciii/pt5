ROOT=$(dirname "$(realpath "$BASH_SOURCE")")

echo "root: $ROOT"

alias blender="blender $ROOT/scene/scene.blend"
alias run="source $ROOT/runall.sh"