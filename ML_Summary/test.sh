#!/bin/bash
for ((i=1;i<21;i++))
do 
mkdir  obj$i
cp obj$i*.png obj$i
done 
echo "success"
