#!/bin/bash

for i in `ls m*.png`; do

    convert -crop 704x476+0+53 $i ${i%%.png}_crop.png

done