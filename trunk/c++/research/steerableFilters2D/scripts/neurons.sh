#!/bin/bash

style=thin
oddOrders=0
order=2
porderl='0 1 3 5'
radiusl='1  3  5  7  9 '


for porder in $porderl; do
    for radius in $radiusl; do
        coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.txt $porder $order $radius $oddOrders $style`
        fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.txt $porder $order $radius $oddOrders $style`
        imageName=`printf N1_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.jpg $porder $order $radius $oddOrders $style`
                # ## coordinates
        /home/ggonzale/workspace/steerableFilters2D/bin/getCoordinatesRadius ../neurons/n1/2/N1_2.jpg $radius ../neurons/n1/n1_cloud_2500_$style.cl ../neurons/n1/coords/$coordsName $order $oddOrders $porder
                # ## Calls the matlab script to do the fisher lds
        dirR=`pwd`
        cd ../matlab/
        matlab -nojvm -r  "fisher ../neurons/n1/coords/$coordsName ../neurons/n1/coords/$fisherName"
        cd $dirR
                # ## steerRadius
        /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../neurons/n1/2/N1_2.jpg $radius ../neurons/n1/2/theta.jpg ../neurons/n1/coords/$fisherName ../neurons/n1/coords/$imageName $order $oddOrders $porder
    done
done


# ## Do the evaluation (play with N7
for porder in $porderl; do
    for radius in $radiusl; do
        rm -rf ../neurons/n7/2/{Linear,Circular,Quadratic,Cubic}
        coordsName=`printf coords_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.txt $porder $order $radius $oddOrders $style`
        fisherName=`printf fisher_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.txt $porder $order $radius $oddOrders $style`
        imageName=`printf N7_porder_%i_order_%i_radius_%i_eOrdsInc_%i_style_%s.jpg $order $porder $radius $oddOrders $style`
        /home/ggonzale/workspace/steerableFilters2D/bin/steerRadius ../neurons/n7/2/N7_2.jpg $radius ../neurons/n7/2/theta.jpg ../neurons/n1/coords/$fisherName ../neurons/n7/coords/$imageName $order $oddOrders $porder

    done
done
