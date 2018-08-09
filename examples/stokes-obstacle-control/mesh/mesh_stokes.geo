lc1 = 0.025;
lc2 = lc1/2;
d =1;
r = 0.126157;
c = d/2;
Point(1) = {0, c, 0, lc1};
Point(2) = {d, c,  0,lc1} ;
Point(3) = {d, d, 0, lc1} ;
Point(4) = {0, d, 0, lc1} ;
Point(5) = {c-r, c, 0, lc2};
Point(6) = {c, c+r, 0, lc2};
Point(7) = {c+r, c, 0, lc2};
Point(8) = {c, c, 0, lc2};


Line(1) = {1,5} ;
Ellipse(2) = {5,8,5,6};
Ellipse(3) = {6,8,6,7};
Line(4) = {7,2} ;
Line(5) = {2,3} ;
Line(6) = {3,4} ;
Line(7) = {4,1} ;
Line Loop(8) = {7,1,2,3,4,5,6} ;
Plane Surface(9) = {8} ;
Symmetry {0,1,0,-c}{Duplicata{Surface{9};}}
Physical Surface(1) = {9};
Physical Surface(2) = {10};


inflow = 1;
outflow = 2;
noslip = 3;
inside = 4;
symmetric = 5;

Physical Line(inflow) = {7,11} ;
Physical Line(outflow) = {5,16} ;
Physical Line(noslip) = {6,17};
Physical Line(inside) = {2,3,13,14};
Physical Line(symmetric) = {1,4};


