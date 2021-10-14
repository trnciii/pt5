RESULT=$(pwd)/result
ROOT=$(dirname "$BASH_SOURCE")


function breakline(){
	echo "---------- ---------- ----------"
}

function bold_magenta(){
	echo -e "\033[35m\033[1m$1\033[0m"
}



echo "ROOT $ROOT"
echo "RESULT $RESULT"

bold_magenta "clean $RESULT"
rm $RESULT/*


bold_magenta "run c++ app"
$ROOT/build/main
breakline


breakline
bold_magenta "run python"
python3 $ROOT/interface/testCore.py
breakline


breakline
bold_magenta "render from blender"
blender $ROOT/scene/scene.blend --python $ROOT/scene/testBScene.py --background
breakline


breakline
echo "listing $RESULT"

ls $RESULT
