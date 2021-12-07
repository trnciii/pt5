ROOT=$(dirname "$(realpath "$BASH_SOURCE")")
RESULT=$(pwd)/result


function breakline(){
	echo ""
	echo "---------- ---------- ----------"
}

# for state
function bold_magenta(){
	echo -e "\033[35m\033[1m$1\033[0m"
}

# for action
function bold_cyan(){
	echo -e "\033[36m\033[1m$1\033[0m"
}



bold_magenta "ROOT   : $ROOT"
bold_magenta "RESULT : $RESULT"

if [ -d "$RESULT" ]; then
	breakline
	bold_cyan "clean $RESULT"
	rm $RESULT/*
fi

breakline
bold_cyan "c++ only"
$ROOT/build/main -o $RESULT/c++


breakline
bold_cyan "python"
python3 $ROOT/interface/testCore.py -o $RESULT/py



breakline
bold_cyan "blender script"
blender $ROOT/scene/scene.blend --python $ROOT/scene/testBScene.py --background -o $RESULT/bl_script


breakline
bold_cyan "blender engine"
blender -b $ROOT/scene/scene.blend -o $RESULT/bl_engine# -f 0


breakline
bold_cyan "listing $RESULT"
ls $RESULT
