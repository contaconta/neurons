#!/bin/bash

oddOrders=0
order=2
porderl='0 1 3 5'
radiusl='1  3  5  7  9 '

## This will clean the computation of the ... are you sure you want that?

## Computes the hessians of the images
# ~/workspace/viva/bin/imageCalculateHessian ../olivier/053/053.png 2.0 ../olivier/053/l1.png ../olivier/053/l2.png 1 ../olivier/053/theta.png
# ~/workspace/viva/bin/imageCalculateHessian ../olivier/136/136.png 2.0 ../olivier/136/l1.png ../olivier/136/l2.png 1 ../olivier/136/theta.png

# exit




## Does the training
mkdir -p ../olivier/053/coords/
mkdir -p ../olivier/136/coords/
for porder in $porderl; do
    for radius in $radiusl; do
        # rm -rf ../drive/d21/{Linear,Circular,Quadratic,Cubic}
        coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
        fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
        imageName=`printf 053_porder_%i_order_%i_radius_%i_eOrdsInc_%i_%s.jpg  $porder $order $radius $oddOrders`
                ## coordinates
        /home/ggonzale/workspace/steerableFilters2D/bin/getCoordinatesRadius ../olivier/053/053.png $radius ../olivier/053/053_2500_2500.cl ../olivier/053/coords/$coordsName $order $oddOrders $porder
                ## Calls the matlab script to do the fisher lds
        dirR=`pwd`
        cd ../matlab/
        matlab -nojvm -r  "fisher ../olivier/053/coords/$coordsName ../olivier/053/coords/$fisherName"
        cd $dirR
                # ## steerRadius
        # /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../olivier/053/053.png $radius ../olivier/053/theta.png ../olivier/053/coords/$fisherName ../olivier/053/coords/$imageName $order $oddOrders $porder
    done
done


## Do the evaluation (play with d22)
for porder in $porderl; do
    for radius in $radiusl; do
        coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
        fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i.txt $porder $order $radius $oddOrders`
        imageName=`printf 136_porder_%i_order_%i_radius_%i_eOrdsInc_%i_%s.jpg  $porder $order $radius $oddOrders`
        /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../olivier/136/136.png $radius ../olivier/136/theta.png ../olivier/053/coords/$fisherName ../olivier/136/coords/$imageName $order $oddOrders $porder
        done
done





































## Do the evaluation (play with 136)
# for radius in  7 9 11; do
    # for order in 2; do
        # rm -rf ../olivier/136/{Linear,Circular,Quadratic,Cubic}
        # # coordsName=`printf coords_order_%i_radius_%i_eOrdsInc_%i.txt $order $radius $oddOrders`
        # # fisherName=`printf fisher_order_%i_radius_%i_eOrdsInc_%i.txt $order $radius $oddOrders`
        # # imageName=`printf 136_order_%i_radius_%i_eOrdsInc_%i_%s.png $order $radius $oddOrders`
        # # # /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../olivier/136/136.png $radius ../olivier/136/theta.png ../olivier/053/coords/$fisherName ../olivier/136/coords/$imageName $order $oddOrders
    # done
# done

# exit


# ## Does the training
# for radius in 5 7 9 11; do
    # for order in 2; do
        # rm -rf ../olivier/053/{Linear,Circular,Quadratic,Cubic}
        # # coordsName=`printf coords_order_%i_radius_%i_eOrdsInc_%i.txt $order $radius $oddOrders`
        # # fisherName=`printf fisher_order_%i_radius_%i_eOrdsInc_%i.txt $order $radius $oddOrders`
        # # imageName=`printf 053_order_%i_radius_%i_eOrdsInc_%i_%s.png $order $radius $oddOrders`
                # # ## coordinates
        # # # # echo /home/ggonzale/workspace/steerableFilters2D/bin/getCoordinatesRadius ../olivier/053/053.png $radius ../olivier/053/053_2500_2500.cl ../olivier/053/coords/$coordsName $order $oddOrders
        # # # /home/ggonzale/workspace/steerableFilters2D/bin/getCoordinatesRadius ../olivier/053/053.png $radius ../olivier/053/053_2500_2500.cl ../olivier/053/coords/$coordsName $order $oddOrders
        # ## Calls the matlab script to do the fisher lds
        # dirR=`pwd`
        # cd ../matlab/
        # # matlab -nojvm -r  "fisher ../olivier/053/coords/$coordsName ../olivier/053/coords/$fisherName"
        # cd $dirR
        # ## steerRadius
        # # # /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../olivier/053/053.png $radius ../olivier/053/theta.png ../olivier/053/coords/$fisherName ../olivier/053/coords/$imageName $order $oddOrders
    # done
# done

