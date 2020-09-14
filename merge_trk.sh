#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

usage() {
	echo "$(basename $0) [ -v ] -o trk_outfile <trk_infile0> <trk_infile1> ..."
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

OUT_FILE=""
VERBOSE=""

OPTS=$(getopt -o "vho:" -- "$@")
eval set -- "$OPTS"

while true; do
	case "$1" in
		-o)
			OUT_FILE=$2
			shift
			shift
			;;
		-v)
			VERBOSE="1"
			shift
			;;
		-h)
			usage
			exit 1
			;;
		--)
			shift
			break
			;;
	esac
done

if [ -z $OUT_FILE ]; then
	echo "Please provide an output file name with the -o option!"
	exit 1
fi

# necessary when running via docker to expand again
# the parameter list turning spaces into separators
set -- $*

TRK_FILES=("$@")
NTRKF=$(($#))

#echo $TRK_FILES
#echo $NTRKF

if [ $VERBOSE ]; then
	echo "Merging $NTRKF files into $OUT_FILE..."
fi

head -c1000 ${TRK_FILES[0]} > $OUT_FILE

NTRACK=0
for((i=0; i<$NTRKF; i++)); do
	if [ $VERBOSE ]; then
		printf "%8d/%8d\r" $i $NTRKF
	fi
	NTRACK=$(($NTRACK + $(od -A none -t dI -j 988 -N4 ${TRK_FILES[$i]})));
	tail -c+1001 ${TRK_FILES[$i]} >> $OUT_FILE
done

NTRACK=$(printf "%08X" $NTRACK)

printf "\x${NTRACK:6:2}\x${NTRACK:4:2}\x${NTRACK:2:2}\x${NTRACK:0:2}" | dd of=$OUT_FILE bs=1 seek=988 count=4 conv=notrunc &> /dev/null
