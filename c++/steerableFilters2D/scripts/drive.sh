#!/bin/bash

oddOrders=0
order=2
porderl='0'
radiusl='1 2 3 4 5 6 7 8 9 10'


## This will clean the computation of the ... are you sure you want that?

## Computes the hessians of the images
# ~/workspace/viva/bin/imageCalculateHessian ../drive/d21/d21_training_green.jpg 2.0 ../drive/d21/l1.jpg ../drive/d21/l2.jpg 1 ../drive/d21/theta.jpg
# ~/workspace/viva/bin/imageCalculateHessian ../drive/d22/d22_training_green.jpg 2.0 ../drive/d22/l1.jpg ../drive/d22/l2.jpg 1 ../drive/d22/theta.jpg


## Does the training
mkdir -p ../drive/d21/coords/
mkdir -p ../drive/d22/coords/
for porder in $porderl; do
    for radius in $radiusl; do
        # rm -rf ../drive/d21/{Linear,Circular,Quadratic,Cubic}
            coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
            fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
            imageName=`printf d21_porder_%i_order_%i_radius_%i_eOrdsInc_%i_%s.jpg  $porder $order $radius $oddOrders`
                ## coordinates
            /home/ggonzale/workspace/steerableFilters2D/bin/getCoordinatesRadius ../drive/d21/d21_training_green.jpg $radius ../drive/d21/d21_5000_5000_0.cl ../drive/d21/coords/$coordsName $order $oddOrders $porder
                ## Calls the matlab script to do the fisher lds
            dirR=`pwd`
            cd ../matlab/
            matlab -nojvm -r  "fisher ../drive/d21/coords/$coordsName ../drive/d21/coords/$fisherName"
            cd $dirR
                # ## steerRadius
            # /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../drive/d21/d21_training_green.jpg $radius ../drive/d21/theta.jpg ../drive/d21/coords/$fisherName ../drive/d21/coords/$imageName $order $oddOrders $porder
        done
done


## Do the evaluation (play with d22)
for porder in $porderl; do
    for radius in $radiusl; do
            coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
            fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
            imageName=`printf d22_porder_%i_order_%i_radius_%i_eOrdsInc_%i_%s.jpg  $porder $order $radius $oddOrders`
            /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../drive/d22/d22_training_green.jpg $radius ../drive/d22/theta.jpg ../drive/d21/coords/$fisherName ../drive/d22/coords/$imageName $order $oddOrders $porder
        done
done
