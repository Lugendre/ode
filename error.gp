set terminal png
set output "error.png"
set logscale
set format x "10^{%L}"
set format y "10^{%L}"
set xlabel "step size"
set ylabel "Relative error"
set grid
set xrange [*:*] reverse

plot "error.txt" using 1:2 w lp title "IRK GL 3/6"
