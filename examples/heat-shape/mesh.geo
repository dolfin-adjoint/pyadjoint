lc1 = 0.02;
lc2 = lc1/5;
cx = 0.5;
cy = 0.5;
a = 0.1;
b = 0.30000000000000004;
width = 3*lc1;
L = 1.0;
H = 1.0;

// Outer shell
Point(1) = {0,H/2,0,lc1};
Point(2) = {L,H/2 , 0, lc1};
Point(3) = {L, H, 0, lc1};
Point(4) = {0, H, 0, lc1};



// Inner geo
// Inner circle
Point(6) = {cx,cy,0,lc2};
Point(7) = {cx-b, cy, 0, lc2};
Point(8) = {cx+b, cy, 0, lc2};
Point(9) = {cx, cy+a, 0, lc2};

Point(10) = {cx-b-width, cy, 0, lc1};
Point(11) = {cx, cy+a+width, 0, lc1};
Point(12) = {cx+b+width, cy, 0, lc1};

Line(1) = {1,10};
Line(2) = {12,2};
Line(3) = {3,4};
Line(4) = {4,1};
Line(12) = {2,3};
Line(6) = {10,7};
Ellipse(7) = {7,6,7,9};
Ellipse(8) = {9,6,9,8};

Line(9) = {8,12};
Ellipse(10) = {12,6,12,11};
Ellipse(11) = {11,6,11,10};

Line Loop(10) = {1,6,7,8,9,2,12,3,4};
// Line Loop(5) = {1,2,3,4};

Plane Surface(12) = {10};

Symmetry {0,1,0,-cy}{Duplicata{Surface{12};}}


//Physical stuff
Physical Surface(13) = {12,14};
Physical Surface(14) = {13,14};

bc = 1;
object = 2;

Physical Line(bc) = {12,4,3,20,21,22};
Physical Line(object) = {16,17,8,7};


